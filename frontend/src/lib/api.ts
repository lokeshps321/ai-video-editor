import type {
  BrollAutoApplyResponse,
  BrollConfig,
  BrollSyncResponse,
  BrollSlot,
  BrollSuggestResponse,
  BrollUndoResponse,
  ExportAspectRatio,
  Job,
  JobEvent,
  MediaAsset,
  OperationHistoryItem,
  Project,
  PromptParse,
  SmartReframeResponse,
  TimelineOperation,
  TimelineOperationResponse,
  Transcript,
  TranscriptCutResponse,
  TranscriptEditResponse,
  TranscriptGenerateResponse,
  TranscriptMode,
  VibeAction,
  VibeActionResponse,
} from "../types";

function resolveDefaultApiBase(): string {
  if (typeof window === "undefined") {
    return "http://127.0.0.1:8000";
  }
  const { hostname, origin, protocol } = window.location;
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return `${protocol}//${hostname}:8000`;
  }
  return origin;
}

const configuredApiBase = String(
  import.meta.env.VITE_API_BASE ?? import.meta.env.VITE_API_BASE_URL ?? "",
).trim();
const DEFAULT_API_BASE = resolveDefaultApiBase();
const API_BASE = (configuredApiBase || DEFAULT_API_BASE).replace(/\/+$/, "");
const parsedDefaultTimeoutMs = Number(
  import.meta.env.VITE_REQUEST_TIMEOUT_MS ?? 120000,
);
const REQUEST_TIMEOUT_MS = Number.isFinite(parsedDefaultTimeoutMs)
  ? Math.max(5000, parsedDefaultTimeoutMs)
  : 120000;
const UPLOAD_TIMEOUT_MS = 5 * 60 * 1000;
// High-quality transcription models can take significantly longer on CPU.
const TRANSCRIPT_TIMEOUT_MS = 30 * 60 * 1000;
const ACTION_TIMEOUT_MS = 30 * 60 * 1000;
const WAVEFORM_TIMEOUT_MS = 5 * 60 * 1000;

type TokenGetter = (forceRefresh?: boolean) => Promise<string | null>;

let _getToken: TokenGetter | null = null;
export function setTokenGetter(getter: TokenGetter | null): void {
  _getToken = getter;
}

function extractErrorMessage(payload: unknown): string {
  if (typeof payload === "string" && payload.trim()) {
    return payload.trim();
  }
  if (payload && typeof payload === "object") {
    const maybeRecord = payload as Record<string, unknown>;
    if (typeof maybeRecord.detail === "string" && maybeRecord.detail.trim()) {
      return maybeRecord.detail.trim();
    }
    if (typeof maybeRecord.message === "string" && maybeRecord.message.trim()) {
      return maybeRecord.message.trim();
    }
    return JSON.stringify(payload);
  }
  return "";
}

function parseContentDispositionFilename(
  contentDisposition: string | null,
): string | null {
  if (!contentDisposition) return null;
  const utf8Match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1]).trim();
    } catch {
      return utf8Match[1].trim();
    }
  }
  const basicMatch = contentDisposition.match(/filename="?([^"]+)"?/i);
  return basicMatch?.[1]?.trim() || null;
}

function isExpiredAuthError(message: string): boolean {
  const lowered = message.toLowerCase();
  return (
    lowered.includes("signature has expired") ||
    lowered.includes("token has expired") ||
    lowered.includes("invalid token") ||
    lowered.includes("authorization header")
  );
}

async function request<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = REQUEST_TIMEOUT_MS,
  options?: { requiresAuth?: boolean },
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  const performRequest = async (forceRefresh = false): Promise<Response> => {
    const token = _getToken ? await _getToken(forceRefresh) : null;
    if (options?.requiresAuth !== false && !token) {
      throw new Error(
        "Your sign-in session is not ready. Please wait a moment and try again.",
      );
    }
    const authHeader: Record<string, string> = token
      ? { Authorization: `Bearer ${token}` }
      : {};
    return fetch(`${API_BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        ...(init?.headers as Record<string, string> | undefined),
        ...authHeader,
      },
    });
  };

  const readErrorMessage = async (res: Response): Promise<string> => {
    let message = "";
    const contentType = res.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      try {
        message = extractErrorMessage(await res.json());
      } catch {
        // Ignore JSON parse issues and fall back to text payload below.
      }
    }
    if (!message) {
      const text = await res.text();
      message = text.trim();
    }
    return message;
  };

  try {
    let res = await performRequest(false);
    if (
      options?.requiresAuth !== false &&
      res.status === 401 &&
      _getToken !== null
    ) {
      const firstMessage = await readErrorMessage(res);
      if (isExpiredAuthError(firstMessage)) {
        res = await performRequest(true);
      } else {
        throw new Error(firstMessage || `Request failed with ${res.status}`);
      }
    }
    if (!res.ok) {
      const message = await readErrorMessage(res);
      throw new Error(message || `Request failed with ${res.status}`);
    }
    return (await res.json()) as T;
  } catch (error) {
    const err = error as Error & { name?: string };
    if (err.name === "AbortError") {
      throw new Error(
        `Request timed out after ${Math.round(timeoutMs / 1000)}s. Check backend API at ${API_BASE}.`,
      );
    }
    if (err.message === "Failed to fetch") {
      throw new Error(
        `Backend not reachable at ${API_BASE}. Start FastAPI server and retry.`,
      );
    }
    throw err;
  } finally {
    window.clearTimeout(timeout);
  }
}

export const api = {
  createProject: (name: string): Promise<Project> =>
    request<Project>("/api/v1/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, fps: 30, width: 1080, height: 1920 }),
    }),

  listProjects: (): Promise<Project[]> =>
    request<Project[]>("/api/v1/projects"),

  getProject: (projectId: string): Promise<Project> =>
    request<Project>(`/api/v1/projects/${projectId}`),

  renameProject: (projectId: string, name: string): Promise<Project> =>
    request<Project>(`/api/v1/projects/${projectId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),

  deleteProject: (projectId: string): Promise<{ detail: string }> =>
    request<{ detail: string }>(`/api/v1/projects/${projectId}`, {
      method: "DELETE",
    }),

  listMedia: (projectId: string): Promise<MediaAsset[]> =>
    request<MediaAsset[]>(
      `/api/v1/media?project_id=${encodeURIComponent(projectId)}`,
    ),

  uploadMedia: async (projectId: string, file: File): Promise<MediaAsset> => {
    const formData = new FormData();
    formData.append("project_id", projectId);
    formData.append("file", file);
    return request<MediaAsset>(
      "/api/v1/media/upload",
      {
        method: "POST",
        body: formData,
      },
      UPLOAD_TIMEOUT_MS,
    );
  },

  parsePrompt: (prompt: string): Promise<PromptParse> =>
    request<PromptParse>("/api/v1/prompt/parse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    }),

  applyPrompt: (
    projectId: string,
    prompt: string,
    expectedVersion?: number,
  ): Promise<TimelineOperationResponse> =>
    request(
      `/api/v1/prompt/apply?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          ...(expectedVersion === undefined
            ? {}
            : { expected_version: expectedVersion }),
        }),
      },
    ),

  applyOperations: (
    projectId: string,
    operations: TimelineOperation[],
    expectedVersion?: number,
  ): Promise<TimelineOperationResponse> =>
    request(
      `/api/v1/timeline/operations?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          operations,
          ...(expectedVersion === undefined
            ? {}
            : { expected_version: expectedVersion }),
        }),
      },
    ),

  listOperationHistory: (projectId: string): Promise<OperationHistoryItem[]> =>
    request<OperationHistoryItem[]>(
      `/api/v1/timeline/history?project_id=${encodeURIComponent(projectId)}`,
    ),

  undo: (projectId: string): Promise<Project> =>
    request<Project>(`/api/v1/projects/${projectId}/undo`, { method: "POST" }),

  redo: (projectId: string): Promise<Project> =>
    request<Project>(`/api/v1/projects/${projectId}/redo`, { method: "POST" }),

  getLatestProjectPreview: (projectId: string): Promise<Job | null> =>
    request<Job | null>(
      `/api/v1/projects/${encodeURIComponent(projectId)}/preview`,
    ),

  renderPreview: (
    projectId: string,
    force = false,
    settings?: {
      aspect_ratio?: ExportAspectRatio;
      fps?: 24 | 30 | 60;
      auto_frame?: boolean;
    },
  ): Promise<Job> =>
    request<Job>(
      `/api/v1/render/preview?project_id=${encodeURIComponent(projectId)}${force ? "&force=true" : ""}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          format: "mp4",
          aspect_ratio: settings?.aspect_ratio ?? "9:16",
          resolution: "720p",
          fps: settings?.fps ?? 30,
          quality: "low",
          auto_frame: settings?.auto_frame ?? false,
        }),
      },
    ),

  ingestUrl: (projectId: string, url: string): Promise<Job> =>
    request<Job>(
      `/api/v1/ingest/url?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      },
    ),

  renderExport: (
    projectId: string,
    settings: {
      format: "mp4" | "mov" | "webm";
      aspect_ratio: ExportAspectRatio;
      resolution: "720p" | "1080p" | "4k";
      fps: 24 | 30 | 60;
      quality: "low" | "medium" | "high" | "max";
      bitrate?: string;
      auto_frame?: boolean;
    },
  ): Promise<Job> =>
    request<Job>(
      `/api/v1/render/export?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      },
    ),

  getJob: (jobId: string): Promise<Job> =>
    request<Job>(`/api/v1/jobs/${jobId}`),
  getJobEvents: (jobId: string): Promise<JobEvent[]> =>
    request<JobEvent[]>(`/api/v1/jobs/${jobId}/events`),
  downloadJobOutput: async (
    jobId: string,
    fallbackFilename = "export.mp4",
  ): Promise<string> => {
    const controller = new AbortController();
    const timeout = window.setTimeout(
      () => controller.abort(),
      REQUEST_TIMEOUT_MS,
    );
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/jobs/${encodeURIComponent(jobId)}/download`,
        {
          signal: controller.signal,
        },
      );
      if (!res.ok) {
        let message = "";
        const contentType = res.headers.get("content-type") ?? "";
        if (contentType.includes("application/json")) {
          try {
            message = extractErrorMessage(await res.json());
          } catch {
            // Ignore JSON parse issues and fall back to text payload below.
          }
        }
        if (!message) {
          message = (await res.text()).trim();
        }
        throw new Error(message || `Download failed with ${res.status}`);
      }

      const blob = await res.blob();
      const filename =
        parseContentDispositionFilename(
          res.headers.get("content-disposition"),
        ) || fallbackFilename;
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = filename;
      link.style.display = "none";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => window.URL.revokeObjectURL(downloadUrl), 0);
      return filename;
    } catch (error) {
      const err = error as Error & { name?: string };
      if (err.name === "AbortError") {
        throw new Error(
          `Download timed out after ${Math.round(REQUEST_TIMEOUT_MS / 1000)}s. Check backend API at ${API_BASE}.`,
        );
      }
      if (err.message === "Failed to fetch") {
        throw new Error(
          `Backend not reachable at ${API_BASE}. Start FastAPI server and retry.`,
        );
      }
      throw err;
    } finally {
      window.clearTimeout(timeout);
    }
  },

  generateTranscript: (
    projectId: string,
    assetId: string,
    mode: TranscriptMode,
    language?: string,
    prompt?: string,
    translateToEnglish?: boolean,
  ): Promise<TranscriptGenerateResponse> =>
    request<TranscriptGenerateResponse>(
      `/api/v1/transcript/generate?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_id: assetId,
          mode,
          ...(language ? { language } : {}),
          ...(prompt ? { prompt } : {}),
          ...(translateToEnglish ? { translate_to_english: true } : {}),
        }),
      },
      TRANSCRIPT_TIMEOUT_MS,
    ),

  generateTranscriptAsync: (
    projectId: string,
    assetId: string,
    mode: TranscriptMode,
    language?: string,
    prompt?: string,
    translateToEnglish?: boolean,
    options?: { forceRegenerate?: boolean },
  ): Promise<Job> => {
    const forceRegenerate = !!options?.forceRegenerate;
    const query = new URLSearchParams({ project_id: projectId });
    if (forceRegenerate) {
      query.set("force", "true");
    }
    return request<Job>(
      `/api/v1/transcript/generate/async?${query.toString()}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_id: assetId,
          mode,
          force_regenerate: forceRegenerate,
          ...(language ? { language } : {}),
          ...(prompt ? { prompt } : {}),
          ...(translateToEnglish ? { translate_to_english: true } : {}),
        }),
      },
      TRANSCRIPT_TIMEOUT_MS,
    );
  },

  getTranscriptGenerateResult: (
    projectId: string,
    jobId: string,
  ): Promise<TranscriptGenerateResponse> =>
    request<TranscriptGenerateResponse>(
      `/api/v1/transcript/generate/results/${encodeURIComponent(jobId)}?project_id=${encodeURIComponent(projectId)}`,
      undefined,
      TRANSCRIPT_TIMEOUT_MS,
    ),

  getTranscript: (
    projectId: string,
    transcriptId?: string,
  ): Promise<Transcript> =>
    request<Transcript>(
      `/api/v1/transcript?project_id=${encodeURIComponent(projectId)}${
        transcriptId ? `&transcript_id=${encodeURIComponent(transcriptId)}` : ""
      }`,
    ),

  applyTranscriptCut: (
    projectId: string,
    transcriptId: string,
    keptWordIds: string[],
    options?: {
      contextSec?: number;
      mergeGapSec?: number;
      minRemovedSec?: number;
    },
  ): Promise<TranscriptCutResponse> =>
    request<TranscriptCutResponse>(
      `/api/v1/transcript/cut?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript_id: transcriptId,
          kept_word_ids: keptWordIds,
          ...(options?.contextSec !== undefined
            ? { context_sec: options.contextSec }
            : {}),
          ...(options?.mergeGapSec !== undefined
            ? { merge_gap_sec: options.mergeGapSec }
            : {}),
          ...(options?.minRemovedSec !== undefined
            ? { min_removed_sec: options.minRemovedSec }
            : {}),
        }),
      },
    ),

  applyVibeAction: (
    projectId: string,
    action: VibeAction,
    assetId?: string,
    options?: Record<string, unknown>,
  ): Promise<VibeActionResponse> =>
    request<VibeActionResponse>(
      `/api/v1/vibe/apply?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          asset_id: assetId,
          options: options ?? {},
        }),
      },
      ACTION_TIMEOUT_MS,
    ),

  smartReframe: (
    projectId: string,
    clipIds?: string[],
  ): Promise<SmartReframeResponse> =>
    request<SmartReframeResponse>(
      `/api/v1/timeline/smart-reframe?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clip_ids: clipIds ?? [] }),
      },
      ACTION_TIMEOUT_MS,
    ),

  suggestBroll: (
    projectId: string,
    payload?: {
      transcript_id?: string;
      max_slots?: number;
      candidates_per_slot?: number;
      min_chunk_words?: number;
      replace_existing?: boolean;
      include_project_assets?: boolean;
      include_external_sources?: boolean;
      ai_rerank?: boolean;
      anchor_word_ids?: string[];
      concept_override?: string;
    },
  ): Promise<BrollSuggestResponse> =>
    request<BrollSuggestResponse>(
      `/api/v1/broll/suggest?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload ?? {}),
      },
    ),

  suggestBrollAsync: (
    projectId: string,
    payload?: {
      transcript_id?: string;
      max_slots?: number;
      candidates_per_slot?: number;
      min_chunk_words?: number;
      replace_existing?: boolean;
      include_project_assets?: boolean;
      include_external_sources?: boolean;
      ai_rerank?: boolean;
      anchor_word_ids?: string[];
      concept_override?: string;
    },
    force = false,
  ): Promise<Job> =>
    request<Job>(
      `/api/v1/broll/suggest/async?project_id=${encodeURIComponent(projectId)}${force ? "&force=true" : ""}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload ?? {}),
      },
    ),

  getSuggestBrollResult: (
    projectId: string,
    jobId: string,
  ): Promise<BrollSuggestResponse> =>
    request<BrollSuggestResponse>(
      `/api/v1/broll/suggest/results/${encodeURIComponent(jobId)}?project_id=${encodeURIComponent(projectId)}`,
    ),

  autoApplyBroll: (
    projectId: string,
    payload?: {
      transcript_id?: string;
      max_slots?: number;
      candidates_per_slot?: number;
      min_chunk_words?: number;
      replace_existing?: boolean;
      include_project_assets?: boolean;
      include_external_sources?: boolean;
      ai_rerank?: boolean;
      clear_existing_overlay?: boolean;
      fallback_to_top_candidate?: boolean;
      min_confidence?: number;
      overlay_opacity?: number;
    },
  ): Promise<BrollAutoApplyResponse> =>
    request<BrollAutoApplyResponse>(
      `/api/v1/broll/auto-apply?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload ?? {}),
      },
      ACTION_TIMEOUT_MS,
    ),

  syncBroll: (
    projectId: string,
    payload?: {
      transcript_id?: string;
      clear_existing_overlay?: boolean;
      overlay_opacity?: number;
      slot_ids?: string[];
    },
  ): Promise<BrollSyncResponse> =>
    request<BrollSyncResponse>(
      `/api/v1/broll/sync?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload ?? {}),
      },
      ACTION_TIMEOUT_MS,
    ),

  undoBrollTransaction: (projectId: string): Promise<BrollUndoResponse> =>
    request<BrollUndoResponse>(
      `/api/v1/broll/undo?project_id=${encodeURIComponent(projectId)}`,
      { method: "POST" },
    ),

  getBrollConfig: (): Promise<BrollConfig> =>
    request<BrollConfig>("/api/v1/broll/config"),

  listBrollSlots: (
    projectId: string,
    transcriptId?: string,
  ): Promise<BrollSlot[]> =>
    request<BrollSlot[]>(
      `/api/v1/broll/slots?project_id=${encodeURIComponent(projectId)}${transcriptId ? `&transcript_id=${encodeURIComponent(transcriptId)}` : ""}`,
    ),

  chooseBrollCandidate: (
    projectId: string,
    slotId: string,
    candidateId: string,
  ): Promise<BrollSlot> =>
    request<BrollSlot>(
      `/api/v1/broll/slots/${encodeURIComponent(slotId)}/choose?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ candidate_id: candidateId }),
      },
    ),

  rerollBrollSlot: (
    projectId: string,
    slotId: string,
    payload?: {
      candidates_per_slot?: number;
      include_project_assets?: boolean;
      include_external_sources?: boolean;
      ai_rerank?: boolean;
      english_gloss_override?: string;
    },
  ): Promise<BrollSlot> =>
    request<BrollSlot>(
      `/api/v1/broll/slots/${encodeURIComponent(slotId)}/reroll?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload ?? {}),
      },
      ACTION_TIMEOUT_MS,
    ),

  rejectBrollSlot: (
    projectId: string,
    slotId: string,
    reason?: string,
  ): Promise<BrollSlot> =>
    request<BrollSlot>(
      `/api/v1/broll/slots/${encodeURIComponent(slotId)}/reject?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: reason ?? "" }),
      },
    ),

  updateWordText: (
    transcriptId: string,
    wordId: string,
    newText: string,
    projectId: string,
  ): Promise<TranscriptEditResponse> =>
    request<TranscriptEditResponse>(
      `/api/v1/transcript/${encodeURIComponent(transcriptId)}/words/${encodeURIComponent(wordId)}?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: newText }),
      },
    ),

  updateTranscriptRange: (
    transcriptId: string,
    projectId: string,
    payload: {
      start_word_id: string;
      end_word_id: string;
      text?: string;
      mode?: "replace" | "blank" | "preserve" | "delete";
    },
  ): Promise<TranscriptEditResponse> =>
    request<TranscriptEditResponse>(
      `/api/v1/transcript/${encodeURIComponent(transcriptId)}/range?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
    ),

  restoreTranscriptSnapshot: (
    projectId: string,
    transcriptId: string,
    payload: {
      words: Transcript["words"];
      timeline: Project["timeline"];
    },
  ): Promise<TranscriptEditResponse> =>
    request<TranscriptEditResponse>(
      `/api/v1/transcript/${encodeURIComponent(transcriptId)}/restore?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
    ),

  getWaveform: (
    assetId: string,
    numPeaks = 800,
  ): Promise<{
    asset_id: string;
    num_peaks: number;
    duration_sec: number;
    peaks: number[];
  }> =>
    request(
      `/api/v1/media/${encodeURIComponent(assetId)}/waveform?num_peaks=${numPeaks}`,
      undefined,
      WAVEFORM_TIMEOUT_MS,
    ),

  health: (): Promise<{ status: string; ffmpeg?: string; ffprobe?: string }> =>
    request("/health", undefined, REQUEST_TIMEOUT_MS, { requiresAuth: false }),

  mediaThumbnailUrl: (
    assetId: string,
    timeSec: number,
    width = 160,
  ): string => {
    const t = Math.max(0, Number.isFinite(timeSec) ? timeSec : 0);
    return `${API_BASE}/api/v1/media/${encodeURIComponent(assetId)}/thumbnail?t=${t.toFixed(2)}&w=${Math.round(width)}`;
  },

  apiBase: API_BASE,
};
