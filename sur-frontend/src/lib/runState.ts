// Pure derivations from backend state for the screens that show a run.
// Kept out of App.tsx so they are unit-tested: every one of these replaced a
// UI behaviour that contradicted what the backend was actually doing.
import { PIPELINE_STAGE_ORDER, type PipelineStageKey, type ProjectRead } from "./types";
import type { StageState } from "./useProjectEvents";

export type RunVerdict =
  | { kind: "idle" }
  | { kind: "active"; stage: string | null }
  | { kind: "stalled"; reason: string }
  | { kind: "awaiting_language" }
  | { kind: "failed"; permanent: boolean; message: string; stage: string | null }
  | { kind: "ready" };

/** One answer to "what is this run doing", from the backend's own fields.
 *  `stalled` comes from worker heartbeats; the UI never runs its own timer. */
export function runVerdict(p: ProjectRead): RunVerdict {
  switch (p.status) {
    case "failed":
      return { kind: "failed", permanent: !!p.error_is_permanent, message: p.error_message ?? "The run failed.", stage: p.current_stage };
    case "ready":
      return { kind: "ready" };
    case "awaiting_language_confirmation":
      return { kind: "awaiting_language" };
    case "queued":
    case "processing":
      return p.stalled
        ? { kind: "stalled", reason: p.stalled_reason ?? "The run stopped reporting progress." }
        : { kind: "active", stage: p.current_stage };
    default:
      return { kind: "idle" };
  }
}

/** A project that may be (re)started with POST /process: failed, or
 *  queued/processing but stalled. Anything else the backend answers 409. */
export function canRestart(p: ProjectRead): boolean {
  const v = runVerdict(p);
  return v.kind === "failed" || v.kind === "stalled";
}

/** Fraction of the pipeline behind this project, from its current stage.
 *  The dashboard used to draw fixed 15%/60% bars regardless of progress. */
export function runProgress(p: ProjectRead): number {
  if (p.status === "ready") return 1;
  if (p.status === "awaiting_language_confirmation") return (PIPELINE_STAGE_ORDER.indexOf("transcribe") + 1) / PIPELINE_STAGE_ORDER.length;
  const i = PIPELINE_STAGE_ORDER.indexOf(p.current_stage as PipelineStageKey);
  if (i < 0) return p.status === "queued" ? 0 : 0;
  return i / PIPELINE_STAGE_ORDER.length;
}

/** Merge live WebSocket stage state with what the project row says has
 *  already happened. Opening the Processing screen mid-run (or after a
 *  reload, when the socket has seen no events) used to show every stage as
 *  not started. Live events win whenever they exist. */
export function seedStages(p: ProjectRead | null, live: StageState[]): StageState[] {
  if (!p) return live;
  const current = PIPELINE_STAGE_ORDER.indexOf(p.current_stage as PipelineStageKey);
  const passed =
    p.status === "ready" ? PIPELINE_STAGE_ORDER.length
    : p.status === "awaiting_language_confirmation" ? PIPELINE_STAGE_ORDER.indexOf("transcribe") + 1
    : current;
  return live.map((s, i) => {
    if (s.done || s.active) return s;
    if (passed >= 0 && i < passed) return { ...s, done: true, progress: 1 };
    if (i === current && (p.status === "processing" || p.status === "queued")) return { ...s, active: true };
    return s;
  });
}
