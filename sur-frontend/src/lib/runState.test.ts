import { describe, expect, it } from "vitest";
import { canRestart, runProgress, runVerdict, seedStages } from "./runState";
import type { ProjectRead } from "./types";
import { PIPELINE_STAGE_ORDER } from "./types";
import type { StageState } from "./useProjectEvents";

function project(over: Partial<ProjectRead>): ProjectRead {
  return {
    id: "p", title: "t", target_languages: ["te"], status: "processing", current_stage: null,
    preserve_emotion: true, clone_voice: false, lip_sync_aware: false, tts_model: null, error_message: null,
    error_is_permanent: null, source_language: null, review_language: true, created_at: "", updated_at: "",
    last_activity_at: null, stalled: false, stalled_reason: null, detected_source_language: null,
    detected_source_language_confidence: null, ...over,
  };
}

const blank = (): StageState[] => PIPELINE_STAGE_ORDER.map((key) => ({
  key, done: false, active: false, progress: 0, completed: null, total: null, eta: null, startedAt: null,
}));

describe("runVerdict", () => {
  it("shows a stalled run as stalled with the backend's reason, not as processing forever", () => {
    const v = runVerdict(project({ status: "queued", stalled: true, stalled_reason: "No worker has picked this run up in 7 min." }));
    expect(v).toEqual({ kind: "stalled", reason: "No worker has picked this run up in 7 min." });
  });

  it("keeps a live run active", () => {
    expect(runVerdict(project({ status: "processing", current_stage: "synthesize" }))).toEqual({ kind: "active", stage: "synthesize" });
  });

  it("reports permanence of a failure", () => {
    expect(runVerdict(project({ status: "failed", error_message: "boom", error_is_permanent: true, current_stage: "translate" })))
      .toEqual({ kind: "failed", permanent: true, message: "boom", stage: "translate" });
  });
});

describe("canRestart matches the backend's /process guard", () => {
  it.each([
    [project({ status: "failed" }), true],
    [project({ status: "processing", stalled: true }), true],
    [project({ status: "queued", stalled: true }), true],
    [project({ status: "processing" }), false],
    [project({ status: "awaiting_language_confirmation" }), false],
    [project({ status: "ready" }), false],
  ])("%#", (p, expected) => {
    expect(canRestart(p)).toBe(expected);
  });
});

describe("runProgress", () => {
  it("reflects the real stage instead of fixed percentages", () => {
    expect(runProgress(project({ status: "queued" }))).toBe(0);
    expect(runProgress(project({ current_stage: "translate" }))).toBeCloseTo(4 / 7);
    expect(runProgress(project({ status: "awaiting_language_confirmation" }))).toBeCloseTo(3 / 7);
    expect(runProgress(project({ status: "ready" }))).toBe(1);
  });
});

describe("seedStages", () => {
  it("marks earlier stages done and the current one active when no live events have arrived", () => {
    const stages = seedStages(project({ current_stage: "translate" }), blank());
    expect(stages.filter((s) => s.done).map((s) => s.key)).toEqual(["extract_audio", "chunk_and_diarize", "transcribe", "detect_emotion"]);
    expect(stages.find((s) => s.active)?.key).toBe("translate");
  });

  it("lets live events win", () => {
    const live = blank().map((s) => (s.key === "synthesize" ? { ...s, active: true, progress: 0.5 } : s));
    const stages = seedStages(project({ current_stage: "translate" }), live);
    expect(stages.find((s) => s.key === "synthesize")?.progress).toBe(0.5);
  });

  it("shows everything done for a ready project", () => {
    expect(seedStages(project({ status: "ready" }), blank()).every((s) => s.done)).toBe(true);
  });
});
