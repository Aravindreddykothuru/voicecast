/**
 * Projects: table/grid toggle, filters, sort, search, and bulk selection.
 *
 * Bulk delete is deliberately NOT wired to a fake action. This API has no
 * DELETE /api/projects/{id}: the button explains that and stays disabled
 * rather than appearing to work, or worse, hiding rows locally so they come
 * back on reload. The endpoint is listed in the report.
 */
import { LayoutGrid, Rows3, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";

import { ProjectCard, stageLabel } from "@/components/ProjectCard";
import { getExport, resolveUrl, retryProject } from "@/lib/api";
import { languageName, sourceLanguageName, useReadyCapabilities } from "@/lib/capabilities";
import { canRestart, runVerdict } from "@/lib/runState";
import type { ProjectListItem, ProjectStatus } from "@/lib/types";
import {
  Button,
  Callout,
  Card,
  ConfirmDialog,
  EmptyState,
  Select,
  Skeleton,
  StatusBadge,
  fmtDuration,
  fmtWhen,
  statusTone,
  useToast,
} from "@/ui";
import type { Screen } from "@/components/AppShell";

const STATUSES: (ProjectStatus | "all")[] = [
  "all", "ready", "processing", "queued", "awaiting_language_confirmation", "failed", "draft", "uploading",
];
type SortKey = "updated" | "created" | "title" | "duration";

export function Projects({
  projects,
  error,
  reload,
  openProject,
  go,
  search,
}: {
  projects: ProjectListItem[] | null;
  error: string | null;
  reload: () => void;
  openProject: (id: string, to: Screen) => void;
  go: (s: Screen) => void;
  search: string;
}) {
  const caps = useReadyCapabilities();
  const toast = useToast();

  const [view, setView] = useState<"grid" | "table">("table");
  const [status, setStatus] = useState<ProjectStatus | "all">("all");
  const [lang, setLang] = useState("all");
  const [sort, setSort] = useState<SortKey>("updated");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirming, setConfirming] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);

  const all = projects ?? [];
  const loading = projects === null && !error;

  const rows = useMemo(() => {
    const q = search.trim().toLowerCase();
    const filtered = all.filter((p) => {
      if (status !== "all" && p.status !== status) return false;
      if (lang !== "all" && !p.target_languages.includes(lang)) return false;
      if (q && !p.title.toLowerCase().includes(q)) return false;
      return true;
    });
    const by: Record<SortKey, (a: ProjectListItem, b: ProjectListItem) => number> = {
      updated: (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
      created: (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      title: (a, b) => a.title.localeCompare(b.title),
      duration: (a, b) => (b.source_video_duration_ms ?? 0) - (a.source_video_duration_ms ?? 0),
    };
    return [...filtered].sort(by[sort]);
  }, [all, status, lang, search, sort]);

  const openTo = (p: ProjectListItem): Screen =>
    ["processing", "queued", "awaiting_language_confirmation", "uploading", "failed", "draft"].includes(p.status)
      ? "processing"
      : "editor";

  const retry = async (id: string) => {
    setBusyId(id);
    try {
      await retryProject(id);
      toast("running", "Restarted.");
      reload();
    } catch (e) {
      toast("danger", e instanceof Error ? e.message : "Retry failed");
    } finally {
      setBusyId(null);
    }
  };

  const download = async (id: string) => {
    setBusyId(id);
    try {
      const exp = await getExport(id);
      if (exp?.output_url) window.open(resolveUrl(exp.output_url), "_blank");
      else toast("warning", "No finished export for that project yet.");
    } catch (e) {
      toast("danger", e instanceof Error ? e.message : "Export lookup failed");
    } finally {
      setBusyId(null);
    }
  };

  const allChecked = rows.length > 0 && rows.every((p) => selected.has(p.id));

  return (
    <div className="p-5 md:p-7 flex flex-col gap-5 max-w-[1700px] mx-auto w-full">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-[26px] font-semibold" style={{ color: "var(--text)" }}>Projects</h1>
          <p className="text-[13px] mt-1" style={{ color: "var(--text-muted)" }}>
            {loading ? "Loading…" : `${rows.length} of ${all.length}`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg overflow-hidden" style={{ border: "1px solid var(--border)" }}>
            {([["table", Rows3], ["grid", LayoutGrid]] as const).map(([v, Icon]) => (
              <button
                key={v}
                onClick={() => setView(v)}
                aria-pressed={view === v}
                aria-label={`${v} view`}
                className="w-9 h-9 flex items-center justify-center"
                style={{
                  background: view === v ? "var(--accent-soft)" : "transparent",
                  color: view === v ? "var(--accent)" : "var(--text-muted)",
                }}
              >
                <Icon size={15} />
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Filters */}
      <Card className="p-3 flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label htmlFor="f-status" className="text-[11px]" style={{ color: "var(--text-dim)" }}>Status</label>
          <Select id="f-status" value={status} onChange={(e) => setStatus(e.target.value as ProjectStatus | "all")} className="w-44">
            {STATUSES.map((s) => (
              <option key={s} value={s}>{s === "all" ? "All statuses" : statusTone(s).label}</option>
            ))}
          </Select>
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="f-lang" className="text-[11px]" style={{ color: "var(--text-dim)" }}>Target language</label>
          <Select id="f-lang" value={lang} onChange={(e) => setLang(e.target.value)} className="w-44">
            <option value="all">All languages</option>
            {caps.languages.map((l) => (
              <option key={l.code} value={l.code}>{l.display_name}</option>
            ))}
          </Select>
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="f-sort" className="text-[11px]" style={{ color: "var(--text-dim)" }}>Sort by</label>
          <Select id="f-sort" value={sort} onChange={(e) => setSort(e.target.value as SortKey)} className="w-44">
            <option value="updated">Last updated</option>
            <option value="created">Created</option>
            <option value="title">Title</option>
            <option value="duration">Duration</option>
          </Select>
        </div>
        <div className="flex-1" />
        {selected.size > 0 && (
          <Button variant="danger" icon={<Trash2 size={14} />} onClick={() => setConfirming(true)}>
            Delete {selected.size}
          </Button>
        )}
      </Card>

      {error && (
        <Callout tone="danger" title="Could not load projects" action={<Button size="sm" variant="ghost" onClick={reload}>Retry</Button>}>
          {error}
        </Callout>
      )}

      {loading && <Card className="p-4 flex flex-col gap-2">{Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-10" />)}</Card>}

      {!loading && all.length === 0 && !error && (
        <Card>
          <EmptyState
            title="No projects yet"
            body="Start a dub and it will appear here."
            action={<Button onClick={() => go("new-project")}>Create your first dub</Button>}
          />
        </Card>
      )}

      {!loading && all.length > 0 && rows.length === 0 && (
        <Card><EmptyState title="Nothing matches those filters" body="Try clearing the status or language filter." /></Card>
      )}

      {rows.length > 0 && view === "grid" && (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {rows.map((p) => (
            <ProjectCard
              key={p.id}
              project={p}
              sourceName={p.source_language ? sourceLanguageName(caps, p.source_language) : "auto-detect"}
              targetNames={p.target_languages.map((c) => languageName(caps, c)).join(", ")}
              busy={busyId === p.id}
              onOpen={() => openProject(p.id, openTo(p))}
              onRetry={() => retry(p.id)}
              onDownload={() => download(p.id)}
            />
          ))}
        </div>
      )}

      {rows.length > 0 && view === "table" && (
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left" style={{ minWidth: 860 }}>
              <caption className="sr-only">All projects</caption>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th scope="col" className="px-3 py-2.5 w-10">
                    <input
                      type="checkbox"
                      checked={allChecked}
                      onChange={(e) => setSelected(e.target.checked ? new Set(rows.map((p) => p.id)) : new Set())}
                      aria-label="Select all shown projects"
                    />
                  </th>
                  {["Title", "Languages", "Status", "Duration", "Segments", "Updated", ""].map((h) => (
                    <th key={h} scope="col" className="px-3 py-2.5 text-[11px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-dim)" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((p) => {
                  const verdict = runVerdict(p);
                  const st = statusTone(p.status, verdict.kind === "stalled");
                  return (
                    <tr key={p.id} style={{ borderBottom: "1px solid var(--border)" }}>
                      <td className="px-3 py-2.5">
                        <input
                          type="checkbox"
                          checked={selected.has(p.id)}
                          onChange={(e) =>
                            setSelected((s) => {
                              const n = new Set(s);
                              if (e.target.checked) n.add(p.id); else n.delete(p.id);
                              return n;
                            })
                          }
                          aria-label={`Select ${p.title}`}
                        />
                      </td>
                      <td className="px-3 py-2.5">
                        <button onClick={() => openProject(p.id, openTo(p))} className="text-[13px] font-medium text-left hover:underline" style={{ color: "var(--text)" }}>
                          {p.title}
                        </button>
                        {stageLabel(p.current_stage) && p.status !== "ready" && (
                          <div className="text-[11px]" style={{ color: "var(--text-dim)" }}>{stageLabel(p.current_stage)}</div>
                        )}
                      </td>
                      <td className="px-3 py-2.5 text-[12px]" style={{ color: "var(--text-muted)" }}>
                        {(p.source_language ? sourceLanguageName(caps, p.source_language) : "auto")} → {p.target_languages.map((c) => languageName(caps, c)).join(", ")}
                      </td>
                      <td className="px-3 py-2.5"><StatusBadge tone={st.tone} pulse={st.pulse}>{st.label}</StatusBadge></td>
                      <td className="px-3 py-2.5 text-[12px]" style={{ color: "var(--text-muted)" }}>{fmtDuration(p.source_video_duration_ms)}</td>
                      <td className="px-3 py-2.5 text-[12px]" style={{ color: "var(--text-muted)" }}>{p.segment_count}</td>
                      <td className="px-3 py-2.5 text-[12px] whitespace-nowrap" style={{ color: "var(--text-muted)" }}>{fmtWhen(p.updated_at)}</td>
                      <td className="px-3 py-2.5">
                        <div className="flex gap-1.5 justify-end">
                          {canRestart(p) && (
                            <Button size="sm" variant="ghost" onClick={() => retry(p.id)} disabled={busyId === p.id}>Retry</Button>
                          )}
                          {p.status === "ready" && (
                            <Button size="sm" variant="ghost" onClick={() => download(p.id)} disabled={busyId === p.id}>Export</Button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Bulk delete: confirmed, then honest about the missing endpoint. */}
      <ConfirmDialog
        open={confirming}
        title={`Delete ${selected.size} project${selected.size === 1 ? "" : "s"}?`}
        confirmLabel="I understand"
        tone="danger"
        body={
          <>
            This cannot be done yet. The backend exposes no{" "}
            <code style={{ color: "var(--text)" }}>DELETE /api/projects/{"{id}"}</code>, so there is nothing to call.
            Removing the rows here would only hide them until the next reload, which is worse than refusing.
          </>
        }
        onConfirm={() => { setConfirming(false); setSelected(new Set()); }}
        onCancel={() => setConfirming(false)}
      />
    </div>
  );
}
