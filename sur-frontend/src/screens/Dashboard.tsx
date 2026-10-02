/**
 * Dashboard: four stat cards and a grid of recent projects.
 *
 * Every number is counted from the projects the backend returned. "Minutes
 * dubbed" sums source_video_duration_ms over finished projects only, so it
 * cannot drift above what was actually produced; projects whose duration the
 * backend never recorded are excluded and said so, rather than counted as
 * zero and quietly lowering the total.
 */
import { AlertTriangle, CheckCircle2, Clock, FolderOpen, Loader2, Plus } from "lucide-react";
import { useCallback, useEffect, useState, type ReactNode } from "react";

import { ProjectCard } from "@/components/ProjectCard";
import { getExport, listProjects, resolveUrl, retryProject } from "@/lib/api";
import { languageName, sourceLanguageName, useReadyCapabilities } from "@/lib/capabilities";
import type { ProjectListItem } from "@/lib/types";
import {
  Button,
  Callout,
  Card,
  EmptyState,
  Skeleton,
  useToast,
  type Tone,
} from "@/ui";
import type { Screen } from "@/components/AppShell";

const ACTIVE = ["processing", "queued", "awaiting_language_confirmation", "uploading"];

function StatCard({
  label,
  value,
  sub,
  icon,
  tone,
  loading,
}: {
  label: string;
  value: string;
  sub?: string;
  icon: ReactNode;
  tone: Tone;
  loading: boolean;
}) {
  const fg = `var(--${tone === "neutral" ? "text-muted" : tone})`;
  return (
    <Card className="p-4 flex items-start gap-3">
      <div
        className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
        style={{ background: tone === "neutral" ? "var(--surface-hover)" : `var(--${tone}-soft)`, color: fg }}
        aria-hidden="true"
      >
        {icon}
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-[13px]" style={{ color: "var(--text-muted)" }}>{label}</div>
        {loading ? (
          <Skeleton className="h-7 w-16 mt-1" />
        ) : (
          <div className="text-[28px] font-semibold leading-tight" style={{ color: "var(--text)", fontFamily: "Sora, sans-serif" }}>
            {value}
          </div>
        )}
        {sub && <div className="text-[12px] mt-0.5 truncate" style={{ color: "var(--text-dim)" }}>{sub}</div>}
      </div>
    </Card>
  );
}

export function Dashboard({
  go,
  openProject,
  projects,
  error,
  reload,
  search,
}: {
  go: (s: Screen) => void;
  openProject: (id: string, to: Screen) => void;
  projects: ProjectListItem[] | null;
  error: string | null;
  reload: () => void;
  search: string;
}) {
  const caps = useReadyCapabilities();
  const toast = useToast();
  const [busyId, setBusyId] = useState<string | null>(null);

  const loading = projects === null && !error;
  const all = projects ?? [];

  const running = all.filter((p) => p.status === "processing").length;
  const queued = all.filter((p) => ["queued", "uploading", "awaiting_language_confirmation"].includes(p.status)).length;
  const failed = all.filter((p) => p.status === "failed").length;
  const ready = all.filter((p) => p.status === "ready");

  // Only finished projects count as dubbed, and only those whose duration the
  // backend actually recorded.
  const timed = ready.filter((p) => p.source_video_duration_ms != null);
  const minutes = timed.reduce((sum, p) => sum + (p.source_video_duration_ms ?? 0), 0) / 60000;
  const untimed = ready.length - timed.length;

  const filtered = search.trim()
    ? all.filter((p) => p.title.toLowerCase().includes(search.trim().toLowerCase()))
    : all;
  const recent = [...filtered]
    .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
    .slice(0, 6);

  const retry = useCallback(
    async (id: string) => {
      setBusyId(id);
      try {
        await retryProject(id);
        toast("running", "Restarted. The run resumes from the last finished segment.");
        reload();
      } catch (e) {
        toast("danger", e instanceof Error ? e.message : "Retry failed");
      } finally {
        setBusyId(null);
      }
    },
    [reload, toast],
  );

  const download = useCallback(
    async (id: string) => {
      setBusyId(id);
      try {
        const exp = await getExport(id);
        if (exp?.output_url) window.open(resolveUrl(exp.output_url), "_blank");
        else toast("warning", "This project has no finished export yet.");
      } catch (e) {
        toast("danger", e instanceof Error ? e.message : "Export lookup failed");
      } finally {
        setBusyId(null);
      }
    },
    [toast],
  );

  return (
    <div className="p-5 md:p-7 flex flex-col gap-5 max-w-[1500px] mx-auto w-full">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-[30px] font-semibold" style={{ color: "var(--text)" }}>Dashboard</h1>
          <p className="text-[14px] mt-1" style={{ color: "var(--text-muted)" }}>
            {caps.tts_engine} voices on {caps.device.toUpperCase()}
            {!caps.tts_commercial_use && caps.tts_engine !== "mock" && " · non-commercial licence"}
          </p>
        </div>
        <Button icon={<Plus size={15} />} onClick={() => go("new-project")}>New Dub</Button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total projects"
          value={String(all.length)}
          icon={<FolderOpen size={17} />}
          tone="neutral"
          loading={loading}
        />
        <StatCard
          label="Minutes dubbed"
          value={minutes >= 1 ? minutes.toFixed(0) : minutes > 0 ? minutes.toFixed(1) : "0"}
          sub={untimed > 0 ? `${untimed} finished without a recorded duration` : `${timed.length} finished`}
          icon={<CheckCircle2 size={17} />}
          tone="success"
          loading={loading}
        />
        <StatCard
          label="Jobs running"
          value={String(running)}
          sub={queued > 0 ? `${queued} queued` : "none queued"}
          icon={<Loader2 size={17} />}
          tone="running"
          loading={loading}
        />
        <StatCard
          label="Failed jobs"
          value={String(failed)}
          sub={failed > 0 ? "open one to see why" : "none"}
          icon={<AlertTriangle size={17} />}
          tone={failed > 0 ? "danger" : "neutral"}
          loading={loading}
        />
      </div>

      {error && (
        <Callout tone="danger" title="Could not load projects" action={<Button size="sm" variant="ghost" onClick={reload}>Retry</Button>}>
          {error}
        </Callout>
      )}

      <div className="flex items-center justify-between">
        <h2 className="text-[16px] font-semibold" style={{ color: "var(--text)", fontFamily: "Inter, sans-serif", letterSpacing: 0 }}>
          Recent projects
        </h2>
        {all.length > recent.length && (
          <Button size="sm" variant="ghost" onClick={() => go("projects")}>View all {all.length}</Button>
        )}
      </div>

      {loading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} className="overflow-hidden">
              <Skeleton className="h-24" rounded="rounded-none" />
              <div className="p-4 flex flex-col gap-2">
                <Skeleton className="h-4 w-2/3" />
                <Skeleton className="h-3 w-1/2" />
                <Skeleton className="h-3 w-1/3" />
              </div>
            </Card>
          ))}
        </div>
      )}

      {!loading && all.length === 0 && !error && (
        <Card>
          <EmptyState
            icon={<Plus size={22} />}
            title="Create your first dub"
            body="Upload a video, pick a target language, and the pipeline takes it from speech recognition through to a muxed export."
            action={<Button icon={<Plus size={15} />} onClick={() => go("new-project")}>Create your first dub</Button>}
          />
        </Card>
      )}

      {!loading && all.length > 0 && recent.length === 0 && (
        <Card>
          <EmptyState title="No projects match that search" body={`Nothing titled like “${search}”.`} />
        </Card>
      )}

      {recent.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {recent.map((p) => (
            <ProjectCard
              key={p.id}
              project={p}
              sourceName={p.source_language ? sourceLanguageName(caps, p.source_language) : "auto-detect"}
              targetNames={p.target_languages.map((c) => languageName(caps, c)).join(", ")}
              busy={busyId === p.id}
              onOpen={() =>
                openProject(p.id, ACTIVE.includes(p.status) || p.status === "failed" || p.status === "draft" ? "processing" : "editor")
              }
              onRetry={() => retry(p.id)}
              onDownload={() => download(p.id)}
            />
          ))}
        </div>
      )}

      {caps.tts_voice_warnings.length > 0 && (
        <Callout tone="warning" title="Voice release warnings">
          <ul className="list-disc pl-4 flex flex-col gap-0.5">
            {caps.tts_voice_warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </Callout>
      )}
    </div>
  );
}

export { Clock };
