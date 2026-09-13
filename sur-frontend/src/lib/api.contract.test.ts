/**
 * The UI against the backend's real contract (api-contract/openapi.json,
 * written by sur-backend/scripts/export_openapi.py and kept current by the
 * backend test test_openapi_contract_snapshot.py).
 *
 * Three bugs this exists to catch before a user does:
 *  - the UI sent a field the backend silently ignored (a pinned source
 *    language that FastAPI dropped, a "preserve pauses" option nothing read);
 *  - the UI called an endpoint in a way the backend rejects or no-ops;
 *  - the UI's types promised response fields the backend never sends.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  API_BASE,
  confirmLanguage,
  confirmUpload,
  createProject,
  createUploadUrl,
  getCapabilities,
  getExport,
  getProject,
  listProjects,
  listSegments,
  patchSegment,
  regenerateSegment,
  retryProject,
  startProcessing,
} from "./api";
import type {
  Capabilities,
  ExportRead,
  ProjectListItem,
  ProjectRead,
  SegmentRead,
  UploadUrlResponse,
} from "./types";

type Schema = Record<string, any>;
// Resolved from the project root: under jsdom, import.meta.url is not a
// file: URL. Vitest runs with the package directory as cwd.
const openapi: Schema = JSON.parse(readFileSync(join(process.cwd(), "src", "lib", "api-contract", "openapi.json"), "utf-8"));
const schemas: Record<string, Schema> = openapi.components.schemas;

function deref(s: Schema): Schema {
  if (s.$ref) return schemas[s.$ref.split("/").pop()];
  if (s.allOf?.length === 1) return deref(s.allOf[0]);
  return s;
}

/** Returns a list of problems; empty means `value` satisfies `schema`. */
function validate(value: unknown, schema: Schema, at = "body"): string[] {
  const s = deref(schema);
  if (s.anyOf) {
    const results = s.anyOf.map((opt: Schema) => validate(value, opt, at));
    return results.some((r: string[]) => r.length === 0) ? [] : results[0];
  }
  if (value === null) return s.type === "null" ? [] : [`${at}: null not allowed`];
  if (s.enum && !s.enum.includes(value)) return [`${at}: ${JSON.stringify(value)} not in ${JSON.stringify(s.enum)}`];
  if (s.const !== undefined && value !== s.const) return [`${at}: must be ${JSON.stringify(s.const)}`];
  switch (s.type) {
    case "object": {
      if (typeof value !== "object" || Array.isArray(value)) return [`${at}: expected object`];
      const props = s.properties ?? {};
      const problems: string[] = [];
      for (const key of s.required ?? []) if (!(key in (value as object))) problems.push(`${at}.${key}: required`);
      for (const [key, v] of Object.entries(value as object)) {
        if (!(key in props)) problems.push(`${at}.${key}: not in the backend schema (it would be silently ignored)`);
        else problems.push(...validate(v, props[key], `${at}.${key}`));
      }
      return problems;
    }
    case "array":
      if (!Array.isArray(value)) return [`${at}: expected array`];
      if (s.minItems && value.length < s.minItems) return [`${at}: needs at least ${s.minItems} item(s)`];
      return value.flatMap((v, i) => validate(v, s.items ?? {}, `${at}[${i}]`));
    case "string":
      return typeof value === "string" ? [] : [`${at}: expected string`];
    case "boolean":
      return typeof value === "boolean" ? [] : [`${at}: expected boolean`];
    case "integer":
    case "number":
      return typeof value === "number" ? [] : [`${at}: expected number`];
    default:
      return [];
  }
}

function operationFor(method: string, url: string): Schema | null {
  const path = new URL(url).pathname;
  for (const [template, ops] of Object.entries<Schema>(openapi.paths)) {
    const re = new RegExp("^" + template.replace(/\{[^}]+\}/g, "[^/]+") + "$");
    if (re.test(path) && ops[method.toLowerCase()]) return ops[method.toLowerCase()];
  }
  return null;
}

const PID = "3f1b2c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d";
const SID = "9a8b7c6d-5e4f-4d3c-9b2a-1f0e9d8c7b6a";
const project = { id: PID, status: "failed", preserve_emotion: true, clone_voice: false, lip_sync_aware: false, tts_model: null,
  review_language: false, source_language: "en" };

describe("every request the UI sends exists and matches the backend schema", () => {
  let calls: { url: string; init: RequestInit }[] = [];
  beforeEach(() => {
    calls = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit = {}) => {
      calls.push({ url, init });
      return new Response(JSON.stringify(project), { status: 200, headers: { "Content-Type": "application/json" } });
    }));
  });
  afterEach(() => vi.restoreAllMocks());

  const cases: [string, () => Promise<unknown>][] = [
    ["listProjects", () => listProjects()],
    ["getProject", () => getProject(PID)],
    ["createProject", () => createProject("Film", ["te"])],
    ["createUploadUrl", () => createUploadUrl(PID, "clip.mp4", "video/mp4")],
    ["confirmUpload", () => confirmUpload(PID, SID, 1000)],
    ["startProcessing (autodetect)", () => startProcessing(PID, { preserve_emotion: true, clone_voice: false, lip_sync_aware: false, review_language: true, source_language: null })],
    ["startProcessing (pinned)", () => startProcessing(PID, { preserve_emotion: true, clone_voice: true, lip_sync_aware: false, review_language: false, source_language: "hi" })],
    ["retryProject", () => retryProject(PID)],
    ["confirmLanguage (accept)", () => confirmLanguage(PID)],
    ["confirmLanguage (force re-run)", () => confirmLanguage(PID, "en", { forceRetranscribe: true })],
    ["listSegments", () => listSegments(PID)],
    ["getExport", () => getExport(PID)],
    ["patchSegment", () => patchSegment(SID, { translated_text: "x" })],
    ["regenerateSegment (re-voice)", () => regenerateSegment(SID, ["synthesize"])],
    ["regenerateSegment (re-translate)", () => regenerateSegment(SID, ["translate", "synthesize"])],
    ["getCapabilities", () => getCapabilities()],
  ];

  it.each(cases)("%s", async (_name, call) => {
    await call();
    expect(calls.length).toBeGreaterThan(0);
    for (const { url, init } of calls) {
      const method = init.method ?? "GET";
      expect(url.startsWith(API_BASE)).toBe(true);
      const op = operationFor(method, url);
      expect(op, `${method} ${url} is not an endpoint the backend serves`).not.toBeNull();
      const bodySchema = op!.requestBody?.content?.["application/json"]?.schema;
      if (init.body) {
        expect(bodySchema, `${method} ${url} sends a body the endpoint does not accept`).toBeDefined();
        expect(validate(JSON.parse(String(init.body)), bodySchema)).toEqual([]);
      } else if (op!.requestBody?.required) {
        throw new Error(`${method} ${url} requires a body the UI does not send`);
      }
    }
  });

  it("the validator itself rejects what the backend would ignore or refuse", () => {
    const body = deref({ $ref: "#/components/schemas/ProcessRequest" });
    expect(validate({ preserve_pauses: true }, body)).toContain("body.preserve_pauses: not in the backend schema (it would be silently ignored)");
    expect(validate({ stages: ["translate"] }, deref({ $ref: "#/components/schemas/RegenerateRequest" }))).toEqual([]);
    expect(validate({ stages: ["bogus"] }, deref({ $ref: "#/components/schemas/RegenerateRequest" })).length).toBeGreaterThan(0);
  });
});

// Record<keyof T, true> must list EXACTLY the TypeScript type's keys (a
// missing or extra key is a compile error); the runtime check then requires
// the same set in the backend schema. Together: a field added, removed or
// renamed on either side fails a check.
const PROJECT_READ: Record<keyof ProjectRead, true> = {
  id: true, title: true, target_languages: true, status: true, current_stage: true, preserve_emotion: true,
  clone_voice: true, lip_sync_aware: true, tts_model: true, error_message: true, error_is_permanent: true,
  source_language: true, review_language: true, created_at: true, updated_at: true, last_activity_at: true,
  stalled: true, stalled_reason: true, detected_source_language: true, detected_source_language_confidence: true,
};
const PROJECT_LIST_ITEM: Record<keyof ProjectListItem, true> = { ...PROJECT_READ, source_video_duration_ms: true, segment_count: true };
const SEGMENT_READ: Record<keyof SegmentRead, true> = {
  id: true, project_id: true, speaker_id: true, index: true, start_ms: true, end_ms: true, source_text: true,
  detected_language: true, detected_language_confidence: true, translated_text: true, emotion_label: true,
  emotion_score: true, emotion_overridden: true, source_audio_url: true, tts_audio_url: true, tts_duration_ms: true,
  sync_offset_pct: true, status: true, error_message: true, created_at: true, updated_at: true,
};
const EXPORT_READ: Record<keyof ExportRead, true> = {
  id: true, project_id: true, status: true, format: true, resolution: true, output_url: true, qa_report: true,
  created_at: true, completed_at: true,
};
const CAPABILITIES: Record<keyof Capabilities, true> = {
  languages: true, source_languages: true, emotions: true, providers: true, emotion_confidence_floor: true,
  asr_autodetect: true, device: true, voice_clone_available: true, stall_after_seconds: true, tts_engine: true,
  tts_licenses: true, tts_commercial_use: true, max_upload_mb: true, accepted_formats: true,
};
const UPLOAD_URL: Record<keyof UploadUrlResponse, true> = {
  source_video_id: true, upload_url: true, storage_key: true, method: true, expires_in: true,
};

describe("response types match what the backend actually returns", () => {
  it.each([
    ["ProjectRead", PROJECT_READ],
    ["ProjectListItem", PROJECT_LIST_ITEM],
    ["SegmentRead", SEGMENT_READ],
    ["ExportRead", EXPORT_READ],
    ["CapabilitiesOut", CAPABILITIES],
    ["UploadUrlResponse", UPLOAD_URL],
  ])("%s", (name, keys) => {
    expect(Object.keys(schemas[name].properties).sort()).toEqual(Object.keys(keys).sort());
  });

  it("status and stage enums the UI switches on are the backend's", () => {
    const projectStatus = deref(schemas.ProjectRead.properties.status).enum;
    expect(projectStatus.sort()).toEqual(
      ["awaiting_language_confirmation", "draft", "failed", "processing", "queued", "ready", "uploading"],
    );
    const segmentStatus = deref(schemas.SegmentRead.properties.status).enum;
    expect(segmentStatus.sort()).toEqual(["emotion_detected", "failed", "muxed", "pending", "synthesized", "transcribed", "translated"]);
  });
});
