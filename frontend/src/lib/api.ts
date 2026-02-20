import type {
  BrollAutoApplyResponse,
  BrollSlot,
  BrollSuggestResponse,
  Job,
  JobEvent,
  MediaAsset,
  OperationHistoryItem,
  Project,
  PromptParse,
  Transcript,
  TranscriptCutResponse,
  TranscriptGenerateResponse,
  VibeAction,
  VibeActionResponse
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const REQUEST_TIMEOUT_MS = 30000;
const UPLOAD_TIMEOUT_MS = 5 * 60 * 1000;
// High-quality transcription models can take significantly longer on CPU.
const TRANSCRIPT_TIMEOUT_MS = 30 * 60 * 1000;
const ACTION_TIMEOUT_MS = 30 * 60 * 1000;

async function request<T>(path: string, init?: RequestInit, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `Request failed with ${res.status}`);
    }
    return (await res.json()) as T;
  } catch (error) {
    const err = error as Error & { name?: string };
    if (err.name === "AbortError") {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s. Check backend API at ${API_BASE}.`);
    }
    if (err.message === "Failed to fetch") {
      throw new Error(`Backend not reachable at ${API_BASE}. Start FastAPI server and retry.`);
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
      body: JSON.stringify({ name, fps: 30, width: 1080, height: 1920 })
    }),

  getProject: (projectId: string): Promise<Project> => request<Project>(`/api/v1/projects/${projectId}`),

  listMedia: (projectId: string): Promise<MediaAsset[]> =>
    request<MediaAsset[]>(`/api/v1/media?project_id=${encodeURIComponent(projectId)}`),

  uploadMedia: async (projectId: string, file: File): Promise<MediaAsset> => {
    const formData = new FormData();
    formData.append("project_id", projectId);
    formData.append("file", file);
    return request<MediaAsset>(
      "/api/v1/media/upload",
      {
        method: "POST",
        body: formData
      },
      UPLOAD_TIMEOUT_MS
    );
  },

  parsePrompt: (prompt: string): Promise<PromptParse> =>
    request<PromptParse>("/api/v1/prompt/parse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt })
    }),

  applyPrompt: (projectId: string, prompt: string): Promise<{ timeline: Project["timeline"] }> =>
    request(`/api/v1/prompt/apply?project_id=${encodeURIComponent(projectId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt })
    }),

  applyOperations: (
    projectId: string,
    operations: Array<{ op_type: string; params: Record<string, unknown>; source?: string }>
  ): Promise<{ timeline: Project["timeline"] }> =>
    request(`/api/v1/timeline/operations?project_id=${encodeURIComponent(projectId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ operations })
    }),

  listOperationHistory: (projectId: string): Promise<OperationHistoryItem[]> =>
    request<OperationHistoryItem[]>(`/api/v1/timeline/history?project_id=${encodeURIComponent(projectId)}`),

  undo: (projectId: string): Promise<Project> =>
    request<Project>(`/api/v1/projects/${projectId}/undo`, { method: "POST" }),

  redo: (projectId: string): Promise<Project> =>
    request<Project>(`/api/v1/projects/${projectId}/redo`, { method: "POST" }),

  renderPreview: (projectId: string, force = false): Promise<Job> =>
    request<Job>(
      `/api/v1/render/preview?project_id=${encodeURIComponent(projectId)}${force ? "&force=true" : ""}`,
      { method: "POST" }
    ),

  ingestUrl: (projectId: string, url: string): Promise<Job> =>
    request<Job>(`/api/v1/ingest/url?project_id=${encodeURIComponent(projectId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    }),

  renderExport: (
    projectId: string,
    settings: { format: "mp4" | "mov" | "webm"; resolution: "720p" | "1080p" | "4k"; fps: 24 | 30 | 60; quality: "low" | "medium" | "high" | "max"; bitrate?: string }
  ): Promise<Job> =>
    request<Job>(`/api/v1/render/export?project_id=${encodeURIComponent(projectId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings)
    }),

  getJob: (jobId: string): Promise<Job> => request<Job>(`/api/v1/jobs/${jobId}`),
  getJobEvents: (jobId: string): Promise<JobEvent[]> => request<JobEvent[]>(`/api/v1/jobs/${jobId}/events`),

  generateTranscript: (projectId: string, assetId: string): Promise<TranscriptGenerateResponse> =>
    request<TranscriptGenerateResponse>(
      `/api/v1/transcript/generate?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ asset_id: assetId })
      },
      TRANSCRIPT_TIMEOUT_MS
    ),

  getTranscript: (projectId: string, transcriptId?: string): Promise<Transcript> =>
    request<Transcript>(
      `/api/v1/transcript?project_id=${encodeURIComponent(projectId)}${transcriptId ? `&transcript_id=${encodeURIComponent(transcriptId)}` : ""
      }`
    ),

  applyTranscriptCut: (
    projectId: string,
    transcriptId: string,
    keptWordIds: string[],
    options?: {
      contextSec?: number;
      mergeGapSec?: number;
      minRemovedSec?: number;
    }
  ): Promise<TranscriptCutResponse> =>
    request<TranscriptCutResponse>(`/api/v1/transcript/cut?project_id=${encodeURIComponent(projectId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        transcript_id: transcriptId,
        kept_word_ids: keptWordIds,
        ...(options?.contextSec !== undefined ? { context_sec: options.contextSec } : {}),
        ...(options?.mergeGapSec !== undefined ? { merge_gap_sec: options.mergeGapSec } : {}),
        ...(options?.minRemovedSec !== undefined ? { min_removed_sec: options.minRemovedSec } : {})
      })
    }),

  applyVibeAction: (
    projectId: string,
    action: VibeAction,
    assetId?: string,
    options?: Record<string, unknown>
  ): Promise<VibeActionResponse> =>
    request<VibeActionResponse>(
      `/api/v1/vibe/apply?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, asset_id: assetId, options: options ?? {} })
      },
      ACTION_TIMEOUT_MS
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
    }
  ): Promise<BrollSuggestResponse> =>
    request<BrollSuggestResponse>(`/api/v1/broll/suggest?project_id=${encodeURIComponent(projectId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload ?? {})
    }),

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
    }
  ): Promise<BrollAutoApplyResponse> =>
    request<BrollAutoApplyResponse>(
      `/api/v1/broll/auto-apply?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload ?? {})
      },
      ACTION_TIMEOUT_MS
    ),

  listBrollSlots: (projectId: string, transcriptId?: string): Promise<BrollSlot[]> =>
    request<BrollSlot[]>(
      `/api/v1/broll/slots?project_id=${encodeURIComponent(projectId)}${transcriptId ? `&transcript_id=${encodeURIComponent(transcriptId)}` : ""}`
    ),

  chooseBrollCandidate: (projectId: string, slotId: string, candidateId: string): Promise<BrollSlot> =>
    request<BrollSlot>(
      `/api/v1/broll/slots/${encodeURIComponent(slotId)}/choose?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ candidate_id: candidateId })
      }
    ),

  rerollBrollSlot: (
    projectId: string,
    slotId: string,
    payload?: {
      candidates_per_slot?: number;
      include_project_assets?: boolean;
      include_external_sources?: boolean;
      ai_rerank?: boolean;
    }
  ): Promise<BrollSlot> =>
    request<BrollSlot>(
      `/api/v1/broll/slots/${encodeURIComponent(slotId)}/reroll?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload ?? {})
      },
      ACTION_TIMEOUT_MS
    ),

  rejectBrollSlot: (projectId: string, slotId: string, reason?: string): Promise<BrollSlot> =>
    request<BrollSlot>(
      `/api/v1/broll/slots/${encodeURIComponent(slotId)}/reject?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: reason ?? "" })
      }
    ),

  updateWordText: (
    transcriptId: string,
    wordId: string,
    newText: string,
    projectId: string
  ): Promise<{ ok: boolean }> =>
    request<{ ok: boolean }>(
      `/api/v1/transcript/${encodeURIComponent(transcriptId)}/words/${encodeURIComponent(wordId)}?project_id=${encodeURIComponent(projectId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: newText })
      }
    ),

  getWaveform: (assetId: string, numPeaks = 800): Promise<{ asset_id: string; num_peaks: number; duration_sec: number; peaks: number[] }> =>
    request(`/api/v1/media/${encodeURIComponent(assetId)}/waveform?num_peaks=${numPeaks}`),

  health: (): Promise<{ status: string; ffmpeg?: string; ffprobe?: string }> => request("/health"),

  apiBase: API_BASE
};
