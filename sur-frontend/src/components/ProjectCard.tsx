/**
 * One project, as a card on the dashboard and in the Projects grid.
 *
 * There is no thumbnail endpoint, so instead of shipping a placeholder image
 * that pretends to be a frame of the video, the tile shows the target
 * language and a status-tinted gradient. A fake thumbnail is still fake data.
 */
import { ArrowRight, Download, RotateCcw } from "lucide-react";

import { canRestart, runProgress, runVerdict } from "@/lib/runState";
import type { ProjectListItem } from "@/lib/types";
import { EXTRA_STAGE_LABELS, PIPELINE_STAGE_LABELS } from "@/lib/types";
import { Button, Card, ProgressBar, StatusBadge, fmtDuration, fmtWhen, statusTone } from "@/ui";

export function stageLabel(stage: string | null): string | null {
  if (!stage) return null;
  return (
    PIPELINE_STAGE_LABELS[stage as keyof typeof PIPELINE_STAGE_LABELS] ??
    EXTRA_STAGE_LABELS[stage] ??
    stage
  );
}

export function ProjectCard({
  project,
  sourceName,
  targetNames,
  onOpen,
  onRetry,
  onDownload,
  busy,
}: {
  project: ProjectListItem;
  sourceName: string;
  targetNames: string;
  onOpen: () => void;
  onRetry: () => void;
  onDownload: () => void;
  busy: boolean;
}) {
  const verdict = runVerdict(project);
  const st = statusTone(project.status, verdict.kind === "stalled");
  const stage = stageLabel(project.current_stage);
  const showProgress = ["processing", "queued", "awaiting_language_confirmation"].includes(project.status);
  const pct = runProgress(project);

  return (
    <Card interactive className="flex flex-col overflow-hidden">
      {/* Tile: no thumbnail endpoint exists, so this is a language plate. */}
      <div
        className="h-24 flex items-center justify-center flex-shrink-0 relative"
        style={{
          background: `linear-gradient(135deg, var(--accent-soft), var(--surface-hover))`,
          borderBottom: "1px solid var(--border)",
        }}
      >
        <span
          className="text-[22px] font-semibold tracking-tight"
          style={{ color: "var(--accent)", fontFamily: "Sora, sans-serif" }}
        >
          {targetNames.split(",")[0]?.trim() || "—"}
        </span>
        <span className="absolute top-2.5 right-2.5">
          <StatusBadge tone={st.tone} pulse={st.pulse}>{st.label}</StatusBadge>
        </span>
      </div>

      <div className="flex flex-col gap-2.5 p-4 flex-1">
        <div className="min-w-0">
          <h3
            className="text-[14px] font-semibold truncate"
            style={{ color: "var(--text)", fontFamily: "Inter, sans-serif", letterSpacing: 0 }}
            title={project.title}
          >
            {project.title}
          </h3>
          <p className="text-[12px] truncate mt-0.5" style={{ color: "var(--text-muted)" }}>
            {sourceName} → {targetNames}
          </p>
        </div>

        <div className="flex items-center gap-3 text-[11px] flex-wrap" style={{ color: "var(--text-dim)" }}>
          <span>{fmtDuration(project.source_video_duration_ms)}</span>
          <span>{project.segment_count} segments</span>
          <span>{fmtWhen(project.updated_at)}</span>
        </div>

        {stage && project.status !== "failed" && project.status !== "ready" && (
          <span className="text-[11px]" style={{ color: "var(--running)" }}>{stage}</span>
        )}
        {verdict.kind === "stalled" && (
          <span className="text-[11px]" style={{ color: "var(--warning)" }}>{verdict.reason}</span>
        )}
        {verdict.kind === "failed" && (
          <span className="text-[11px] line-clamp-2" style={{ color: "var(--danger)" }} title={verdict.message}>
            {verdict.message}
          </span>
        )}

        {showProgress && <ProgressBar value={pct} tone={st.tone} height={4} label={`${project.title} progress`} />}

        <div className="flex items-center gap-2 mt-auto pt-1">
          <Button size="sm" variant="secondary" onClick={onOpen} icon={<ArrowRight size={13} />}>
            Open
          </Button>
          {canRestart(project) && (
            <Button size="sm" variant="ghost" onClick={onRetry} disabled={busy} icon={<RotateCcw size={13} />}>
              {verdict.kind === "stalled" ? "Restart" : "Retry"}
            </Button>
          )}
          {project.status === "ready" && (
            <Button size="sm" variant="ghost" onClick={onDownload} disabled={busy} icon={<Download size={13} />}>
              Export
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}
