/**
 * Job progress: the seven stages live, an overall bar with the ETA the
 * backend's own throughput implies, the source-language gate, and the failure
 * path (retry the run, read the log).
 *
 * Live data is the project's WebSocket (/ws/projects/{id}) seeded from the
 * project row, with a 4s poll alongside it because the row carries things the
 * socket does not: status, stalled, error_is_permanent and the detected
 * language. The old screen's hard-won rules are kept:
 *
 *  - `detected_source_language` is the backend's verdict and the only thing
 *    the gate may show. Recounting segments in the browser once named a
 *    different language than "Continue" actually locked in.
 *  - A permanent error must never offer "Retry": it cannot succeed.
 *  - "Re-run ASR" has to send force_retranscribe, or sending the detected
 *    code back is a silent no-op.
 */
import { Activity, ArrowRight, FileText, RotateCcw, Terminal } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { StageTracker } from "@/components/StageTracker";
import { stageLabel } from "@/components/ProjectCard";
import { confirmLanguage, getProject, listSegments, retryProject } from "@/lib/api";
import { sourceLanguageName, useReadyCapabilities } from "@/lib/capabilities";
import { canRestart, runProgress, runVerdict, seedStages } from "@/lib/runState";
import { projectError, type ProjectRead, type SegmentRead } from "@/lib/types";
import { fmtTime, useProjectEvents } from "@/lib/useProjectEvents";
import {
  Button,
  Callout,
  Card,
  CardHeader,
  EmptyState,
  ProgressBar,
  Select,
  Skeleton,
  StatusBadge,
  statusTone,
  useToast,
} from "@/ui";
import type { Screen } from "@/components/AppShell";

export function ProgressScreen({ projectId, go }: { projectId: string | null; go: (s: Screen) => void }) {
  const caps = useReadyCapabilities();
  const toast = useToast();
  const events = useProjectEvents(projectId);

  const [project, setProject] = useState<ProjectRead | null>(null);
  const [segments, setSegments] = useState<SegmentRead[]>([]);
  const [override, setOverride] = useState("");
  const [gateBusy, setGateBusy] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showLog, setShowLog] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(() => {
    if (!projectId) return;
    getProject(projectId).then(setProject).catch((e) => setError(e instanceof Error ? e.message : "Failed to load the project"));
    listSegments(projectId).then(setSegments).catch(() => { /* segments appear after ASR; absence is not an error */ });
  }, [projectId]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    if (showLog && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [events.log, showLog]);

  if (!projectId) {
    return (
      <Card className="m-6">
        <EmptyState title="No project open" body="Open one from the dashboard." action={<Button onClick={() => go("dashboard")}>Go to dashboard</Button>} />
      </Card>
    );
  }

  const perr = project ? projectError(project) : null;
  const verdict = project ? runVerdict(project) : null;
  const awaiting = project?.status === "awaiting_language_confirmation";
  const stages = seedStages(project, events.stages);
  const st = project ? statusTone(project.status, verdict?.kind === "stalled") : null;
  const overall = project ? runProgress(project) : 0;
  const activeStage = stages.find((s) => s.active);

  const detected = project?.detected_source_language
    ? {
        code: project.detected_source_language,
        confidence: project.detected_source_language_confidence ?? 0,
        count: segments.filter((s) => s.detected_language === project.detected_source_language && (s.source_text ?? "").trim()).length,
        spoken: segments.filter((s) => (s.source_text ?? "").trim()).length,
      }
    : null;

  const submitGate = async (rerun: boolean) => {
    setGateBusy(true);
    setError(null);
    try {
      if (rerun) await confirmLanguage(projectId, override || detected?.code, { forceRetranscribe: true });
      else await confirmLanguage(projectId, override || null);
      toast("running", rerun ? "Re-running transcription." : "Language confirmed, the run continues.");
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not confirm the language");
    } finally {
      setGateBusy(false);
    }
  };

  const restart = async () => {
    setRestarting(true);
    setError(null);
    try {
      await retryProject(projectId);
      toast("running", "Restarted from the last finished segment.");
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not restart the run");
    } finally {
      setRestarting(false);
    }
  };

  return (
    <div className="p-5 md:p-7 flex flex-col gap-5 max-w-[1500px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-[30px] font-semibold truncate" style={{ color: "var(--text)" }}>
              {project ? project.title : "Loading…"}
            </h1>
            {st && <StatusBadge tone={st.tone} pulse={st.pulse}>{st.label}</StatusBadge>}
          </div>
          <p className="text-[14px] mt-1" style={{ color: "var(--text-muted)" }}>
            {project
              ? `${project.source_language ? sourceLanguageName(caps, project.source_language) : "auto-detect"} → ${project.target_languages.map((c) => caps.languages.find((l) => l.code === c)?.display_name ?? c).join(", ")}`
              : ""}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[12px] inline-flex items-center gap-1.5" style={{ color: events.connected ? "var(--success)" : "var(--text-dim)" }}>
            <Activity size={13} aria-hidden="true" />
            {events.connected ? "live" : "polling"}
          </span>
          {project?.status === "ready" && (
            <>
              <Button variant="secondary" onClick={() => go("editor")} icon={<ArrowRight size={14} />}>Open studio</Button>
              <Button variant="ghost" onClick={() => go("export")} icon={<FileText size={14} />}>Export</Button>
            </>
          )}
        </div>
      </div>

      {perr && (
        <Callout
          tone={perr.kind === "permanent" ? "danger" : "warning"}
          title={perr.kind === "permanent" ? "This run cannot succeed" : "The run failed"}
          action={
            <div className="flex gap-2 mt-1">
              {perr.kind !== "permanent" && project && canRestart(project) && (
                <Button size="sm" onClick={restart} loading={restarting} icon={<RotateCcw size={13} />}>
                  Retry from the last finished segment
                </Button>
              )}
              <Button size="sm" variant="ghost" onClick={() => setShowLog(true)} icon={<Terminal size={13} />}>View logs</Button>
            </div>
          }
        >
          {perr.message}
          {perr.stage && <span className="block mt-1" style={{ color: "var(--text-muted)" }}>Failed at: {stageLabel(perr.stage)}</span>}
        </Callout>
      )}

      {verdict?.kind === "stalled" && (
        <Callout
          tone="warning"
          title="No progress"
          action={<Button size="sm" onClick={restart} loading={restarting} icon={<RotateCcw size={13} />}>Restart the run</Button>}
        >
          {verdict.reason}
        </Callout>
      )}

      {events.errorMessage && project && project.status !== "failed" && (
        <Callout tone="warning" title={`A stage reported an error${events.errorIsPermanent ? "" : " and will retry"}`}>
          {events.errorMessage}
        </Callout>
      )}

      {project?.current_stage === "regenerate" && verdict?.kind === "active" && (
        <Callout tone="running" title="Regenerating a segment">The export is re-muxed when it finishes.</Callout>
      )}

      {project && (project.status === "draft" || project.status === "uploading") && (
        <Callout tone="neutral" title="Not started">
          This project has not been started{project.status === "uploading" ? " — its upload never completed" : ""}.
        </Callout>
      )}

      {/* Source-language gate */}
      {awaiting && (
        <Card className="p-5 flex flex-col gap-4" style={{ borderColor: "var(--warning-border)" }}>
          <div>
            <h2 className="text-[16px] font-semibold" style={{ color: "var(--warning)", fontFamily: "Inter, sans-serif", letterSpacing: 0 }}>
              Confirm the source language
            </h2>
            <p className="text-[14px] mt-1" style={{ color: "var(--text-muted)" }}>
              The expensive stages have not run yet. Getting this wrong translates the whole video from the wrong
              language.
            </p>
          </div>

          {detected ? (
            <div className="flex items-center gap-3 flex-wrap">
              <StatusBadge tone={detected.confidence < 0.7 ? "warning" : "success"}>
                {sourceLanguageName(caps, detected.code)}
              </StatusBadge>
              <span className="text-[13px]" style={{ color: "var(--text-muted)" }}>
                {(detected.confidence * 100).toFixed(0)}% confidence · {detected.count}/{detected.spoken} spoken segments
              </span>
            </div>
          ) : (
            <span className="text-[14px]" style={{ color: "var(--text-muted)" }}>ASR has not reported a language yet.</span>
          )}

          {detected && detected.confidence < 0.7 && (
            <Callout tone="warning">Low confidence — check this before continuing.</Callout>
          )}

          <Select value={override} onChange={(e) => setOverride(e.target.value)} aria-label="Source language">
            <option value="">{detected ? `Keep detected (${sourceLanguageName(caps, detected.code)})` : "Choose a language"}</option>
            {caps.source_languages.map((l) => (
              <option key={l.code} value={l.code}>{l.display_name}</option>
            ))}
          </Select>

          <div className="flex gap-2 flex-wrap">
            <Button onClick={() => submitGate(false)} loading={gateBusy} disabled={!detected && !override}>Continue</Button>
            <Button variant="ghost" onClick={() => submitGate(true)} disabled={gateBusy || (!override && !detected)}>
              Re-run transcription in {override ? sourceLanguageName(caps, override) : detected ? sourceLanguageName(caps, detected.code) : "this language"}
            </Button>
          </div>
        </Card>
      )}

      {/* Overall + stages */}
      <Card className="p-5 flex flex-col gap-5">
        <div className="flex items-end justify-between gap-4">
          <div>
            <div className="text-[13px]" style={{ color: "var(--text-muted)" }}>Overall</div>
            <div className="text-[26px] font-semibold" style={{ color: "var(--text)", fontFamily: "Sora, sans-serif" }}>
              {Math.round(overall * 100)}%
            </div>
          </div>
          <div className="text-right text-[13px]" style={{ color: "var(--text-muted)" }}>
            {activeStage ? (
              <>
                <div style={{ color: "var(--text)" }}>{stageLabel(activeStage.key)}</div>
                <div>
                  {activeStage.completed != null && activeStage.total != null && `${activeStage.completed}/${activeStage.total}`}
                  {activeStage.eta ? ` · about ${activeStage.eta} left` : ""}
                </div>
              </>
            ) : project?.status === "ready" ? (
              "Finished"
            ) : (
              "Waiting"
            )}
          </div>
        </div>
        <ProgressBar value={overall} tone={st?.tone ?? "running"} label="Overall progress" />
        {project ? (
          <StageTracker stages={stages} failedStage={project.status === "failed" ? project.current_stage : null} />
        ) : (
          <Skeleton className="h-16" />
        )}
      </Card>

      {/* Live log */}
      <Card>
        <CardHeader
          title="Live log"
          sub={events.connected ? "Streaming from the project's event channel" : "Socket not connected; the run is still polled"}
          action={
            <Button size="sm" variant="ghost" onClick={() => setShowLog((v) => !v)} icon={<Terminal size={13} />}>
              {showLog ? "Hide" : "Show"}
            </Button>
          }
        />
        {showLog && (
          <div ref={logRef} className="max-h-72 overflow-y-auto px-5 py-3 flex flex-col gap-1">
            {events.log.length === 0 && (
              <span className="text-[13px]" style={{ color: "var(--text-dim)" }}>No events yet.</span>
            )}
            {events.log.map((l, i) => (
              <div key={i} className="flex gap-3 text-[13px]" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                <span className="flex-shrink-0" style={{ color: "var(--text-dim)" }}>{fmtTime(l.ts)}</span>
                <span
                  className="flex-shrink-0"
                  style={{
                    color:
                      l.level === "success" ? "var(--success)"
                      : l.level === "warn" ? "var(--warning)"
                      : l.level === "error" ? "var(--danger)"
                      : "var(--running)",
                  }}
                >
                  {l.level.toUpperCase().padEnd(7)}
                </span>
                <span style={{ color: "var(--text-mid)" }}>{l.message}</span>
              </div>
            ))}
          </div>
        )}
      </Card>

      {error && <Callout tone="danger">{error}</Callout>}
    </div>
  );
}
