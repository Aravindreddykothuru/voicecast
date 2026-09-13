// REST client for sur-backend (see sur-backend/app/api/routes_*.py). Every
// call goes through `request`, which prefixes VITE_API_BASE_URL and attaches
// the stubbed-auth X-User-Email header (app/core/security.py) so the
// frontend never needs to touch fetch() directly.
import type {
  Capabilities,
  EmotionLabel,
  ExportRead,
  ProjectListItem,
  ProjectRead,
  SegmentRead,
  UploadUrlResponse,
} from "./types";

export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");
export const USER_EMAIL = import.meta.env.VITE_DEV_USER_EMAIL ?? "dev@sur.local";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

/** Resolves a backend-relative URL (e.g. LocalStorage's "/api/storage/files/...")
 *  against API_BASE; leaves absolute URLs (a real S3/MinIO presigned URL) alone. */
export function resolveUrl(url: string): string {
  if (/^https?:\/\//i.test(url)) return url;
  return `${API_BASE}${url.startsWith("/") ? "" : "/"}${url}`;
}

function authHeader(): Record<string, string> {
  // Deliberately re-read from storage on every call rather than cached at
  // module load: a login/logout elsewhere in the app must take effect on
  // the very next request, not after a reload.
  try {
    const raw = localStorage.getItem("sur.auth");
    const token = raw ? (JSON.parse(raw) as { token?: string }).token : null;
    return token ? { Authorization: `Bearer ${token}` } : {};
  } catch {
    return {};
  }
}

/** FastAPI error bodies come in two shapes: HTTPException gives
 *  {detail: "message"}, request validation gives {detail: [{loc, msg}, ...]}.
 *  The second used to be passed straight to `new Error(...)` and rendered as
 *  "[object Object]". Exported for tests. */
export function errorDetail(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) => {
        const field = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
        return field ? `${field}: ${d.msg ?? "invalid"}` : (d.msg ?? "invalid");
      })
      .join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        // Real identity once logged in; the dev stub only matters before
        // that (see app/core/security.py's layered get_current_user).
        "X-User-Email": USER_EMAIL,
        ...authHeader(),
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch {
    // fetch rejects (not resolves with a status) when the API is down or
    // blocked by CORS; the raw TypeError said only "Failed to fetch".
    throw new ApiError(0, `Cannot reach the API at ${API_BASE}. Is the backend running?`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = errorDetail(await res.json(), detail);
    } catch {
      /* non-JSON error body -- keep statusText */
    }
    if (res.status === 401) {
      // The session's token was invalid or expired server-side -- holding
      // onto it would just fail the same way on every retry, so drop it and
      // let CapabilitiesGate/the landing screen ask for a fresh login.
      try {
        localStorage.removeItem("sur.auth");
      } catch {
        /* ignore */
      }
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ── Projects ────────────────────────────────────────────────────────────
export function listProjects(): Promise<ProjectListItem[]> {
  return request("/api/projects");
}

export function createProject(title: string, targetLanguages: string[]): Promise<ProjectRead> {
  return request("/api/projects", {
    method: "POST",
    body: JSON.stringify({ title, target_languages: targetLanguages }),
  });
}

export function getProject(projectId: string): Promise<ProjectRead> {
  return request(`/api/projects/${projectId}`);
}

export function createUploadUrl(
  projectId: string,
  filename: string,
  contentType: string,
): Promise<UploadUrlResponse> {
  return request(`/api/projects/${projectId}/upload`, {
    method: "POST",
    body: JSON.stringify({ filename, content_type: contentType }),
  });
}

/** PUTs the raw file straight to the (possibly relative, local-storage) presigned URL. */
export async function putUploadFile(uploadUrl: string, file: File): Promise<void> {
  const res = await fetch(resolveUrl(uploadUrl), {
    method: "PUT",
    headers: { "Content-Type": file.type || "application/octet-stream" },
    body: file,
  });
  if (!res.ok) throw new ApiError(res.status, `Upload failed: ${res.statusText}`);
}

export function confirmUpload(
  projectId: string,
  sourceVideoId: string,
  durationMs?: number,
): Promise<ProjectRead> {
  return request(`/api/projects/${projectId}/upload/confirm`, {
    method: "POST",
    body: JSON.stringify({ source_video_id: sourceVideoId, duration_ms: durationMs ?? null }),
  });
}

export interface ProcessOptions {
  preserve_emotion: boolean;
  clone_voice: boolean;
  lip_sync_aware: boolean;
  tts_model?: string | null;
  /** Pause after ASR so the detected source language can be confirmed
   *  before the expensive stages run. Backend default is true. */
  review_language?: boolean;
  /** Known source language. ASR decodes in it instead of detecting, and the
   *  backend skips the review gate (there is nothing left to confirm). */
  source_language?: string | null;
}

export function startProcessing(projectId: string, opts: ProcessOptions): Promise<ProjectRead> {
  return request(`/api/projects/${projectId}/process`, {
    method: "POST",
    body: JSON.stringify(opts),
  });
}

/** Re-run the pipeline for a failed project, reusing the flags it was created
 *  with. POST /process has no state guard -- it resets status to queued,
 *  clears error_message and re-enqueues -- so a retry is just re-posting it. */
export async function retryProject(projectId: string): Promise<ProjectRead> {
  const p = await getProject(projectId);
  return startProcessing(projectId, {
    preserve_emotion: p.preserve_emotion,
    clone_voice: p.clone_voice,
    lip_sync_aware: p.lip_sync_aware,
    tts_model: p.tts_model,
    review_language: p.review_language,
    source_language: p.source_language,
  });
}

/** Accept or correct the detected source language, then resume the run.
 *  A code different from the backend's detection re-runs ASR with it;
 *  `forceRetranscribe` re-runs ASR even with the detected language (the
 *  "re-run ASR" button used to send the detected code, which the backend
 *  treated as a plain accept -- a silent no-op). */
export function confirmLanguage(
  projectId: string,
  sourceLanguage?: string | null,
  opts: { forceRetranscribe?: boolean } = {},
): Promise<ProjectRead> {
  return request(`/api/projects/${projectId}/confirm-language`, {
    method: "POST",
    body: JSON.stringify({ source_language: sourceLanguage ?? null, force_retranscribe: !!opts.forceRetranscribe }),
  });
}

export function listSegments(projectId: string): Promise<SegmentRead[]> {
  return request(`/api/projects/${projectId}/segments`);
}

export function getExport(projectId: string): Promise<ExportRead | null> {
  return request<ExportRead>(`/api/projects/${projectId}/export`).catch((e) => {
    if (e instanceof ApiError && e.status === 404) return null;
    throw e;
  });
}

// ── Segments ────────────────────────────────────────────────────────────
export function patchSegment(
  segmentId: string,
  patch: { translated_text?: string; emotion_label?: EmotionLabel },
): Promise<SegmentRead> {
  return request(`/api/segments/${segmentId}`, { method: "PATCH", body: JSON.stringify(patch) });
}

/** "synthesize" re-voices the saved translation (use after editing it);
 *  "translate" + "synthesize" re-derives the text first, discarding edits.
 *  The backend re-muxes the export afterwards and reports progress on the
 *  project (current_stage "regenerate"). Translate-only is rejected: it
 *  would leave the dub saying the old line. */
export type RegenerateStages = ["synthesize"] | ["translate", "synthesize"];

export function regenerateSegment(segmentId: string, stages: RegenerateStages) {
  return request<{ ok: boolean }>(`/api/segments/${segmentId}/regenerate`, {
    method: "POST",
    body: JSON.stringify({ stages }),
  });
}

// NOTE: there is deliberately no emotion label mapping here. The backend's
// labels ARE the labels -- translating them into a fixed UI vocabulary is
// what let the interface show labels the model never predicts.
// Colours and the full label set arrive with /api/capabilities.

// ── Capabilities ────────────────────────────────────────────────────────
export function getCapabilities(): Promise<Capabilities> {
  return request("/api/capabilities");
}

/** Whether the API process answers at all (GET /healthz). The header used to
 *  print a hardcoded "Core: Online" whatever the backend's state. */
export async function apiReachable(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/healthz`);
    return res.ok;
  } catch {
    return false;
  }
}
