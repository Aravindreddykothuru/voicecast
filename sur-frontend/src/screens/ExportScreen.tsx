/**
 * Export: the muxed video, plus subtitles generated in the browser from the
 * segments the API already returns.
 *
 * The video comes from ExportRead.output_url and its size from
 * ExportRead.output_size_bytes -- the backend asks its own storage. The
 * client deliberately does NOT probe with HEAD: when a <video> is streaming
 * that same URL, Chrome reuses the media load's opaque response and the
 * CORS check on the HEAD then fails (reproduced: HEAD alone 200, HEAD during
 * playback "Failed to fetch"), and giving the HEAD its own query string to
 * dodge that would invalidate an S3 presigned signature.
 *
 * SRT and VTT are built here from real segment timings and translated text --
 * that is a format change on data the client already holds, not invented
 * content. Audio-only and per-language files are NOT offered: the backend
 * produces one muxed artefact per project and there is no endpoint for an
 * audio render or a per-language bundle. Those are listed in the report.
 */
import { Download, FileText, RefreshCw, Subtitles } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { getExport, listSegments, resolveUrl } from "@/lib/api";
import type { ExportRead, SegmentQaEntry, SegmentRead } from "@/lib/types";
import {
  Button,
  Callout,
  Card,
  CardHeader,
  EmptyState,
  Skeleton,
  StatusBadge,
  fmtBytes,
  msToTimecode,
  statusTone,
  useToast,
} from "@/ui";
import type { Screen } from "@/components/AppShell";

function srtTime(ms: number): string {
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const msec = Math.floor(ms % 1000);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")},${String(msec).padStart(3, "0")}`;
}

function vttTime(ms: number): string {
  return srtTime(ms).replace(",", ".");
}

function buildSrt(segs: SegmentRead[]): string {
  return segs
    .filter((s) => (s.translated_text ?? "").trim())
    .map((s, i) => `${i + 1}\n${srtTime(s.start_ms)} --> ${srtTime(s.end_ms)}\n${s.translated_text!.trim()}\n`)
    .join("\n");
}

function buildVtt(segs: SegmentRead[]): string {
  const body = segs
    .filter((s) => (s.translated_text ?? "").trim())
    .map((s) => `${vttTime(s.start_ms)} --> ${vttTime(s.end_ms)}\n${s.translated_text!.trim()}\n`)
    .join("\n");
  return `WEBVTT\n\n${body}`;
}

function downloadText(name: string, text: string, mime: string) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export function ExportScreen({ projectId, go }: { projectId: string | null; go: (s: Screen) => void }) {
  const toast = useToast();
  const [exp, setExp] = useState<ExportRead | null>(null);
  const [segs, setSegs] = useState<SegmentRead[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    if (!projectId) return;
    setError(null);
    Promise.all([getExport(projectId), listSegments(projectId).catch(() => [])])
      .then(([e, rows]) => {
        setExp(e);
        setSegs(rows);
        setLoaded(true);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load the export"));
  }, [projectId]);

  useEffect(load, [load]);

  if (!projectId) {
    return (
      <Card className="m-6">
        <EmptyState title="No project open" body="Open one from the dashboard." action={<Button onClick={() => go("dashboard")}>Go to dashboard</Button>} />
      </Card>
    );
  }

  const subtitled = segs.filter((s) => (s.translated_text ?? "").trim()).length;
  const qaSegs: SegmentQaEntry[] = exp?.qa_report?.segments ?? [];
  const st = exp ? statusTone(exp.status) : null;

  return (
    <div className="p-5 md:p-7 flex flex-col gap-5 max-w-[1500px] mx-auto w-full">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-[26px] font-semibold" style={{ color: "var(--text)" }}>Export</h1>
            {st && <StatusBadge tone={st.tone} pulse={st.pulse}>{st.label}</StatusBadge>}
          </div>
          <p className="text-[13px] mt-1" style={{ color: "var(--text-muted)" }}>Final muxed video, subtitles and the QA report</p>
        </div>
        <Button variant="ghost" onClick={load} icon={<RefreshCw size={14} />}>Refresh</Button>
      </div>

      {error && <Callout tone="danger">{error}</Callout>}
      {!loaded && !error && <Skeleton className="h-32" />}

      {loaded && !exp && (
        <Card>
          <EmptyState
            title="No export yet"
            body="One is created automatically when the pipeline finishes muxing."
            action={<Button variant="secondary" onClick={() => go("processing")}>See progress</Button>}
          />
        </Card>
      )}

      {exp && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader title="Dubbed video" sub={`${exp.format} · ${exp.resolution}`} />
            <div className="p-5 flex flex-col gap-3">
              {exp.output_url ? (
                <>
                  <video
                    src={resolveUrl(exp.output_url)}
                    controls
                    className="w-full rounded-lg"
                    style={{ background: "#000", maxHeight: 260 }}
                    aria-label="Dubbed output"
                  />
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-[12px]" style={{ color: "var(--text-muted)" }}>
                      {exp.output_size_bytes != null ? fmtBytes(exp.output_size_bytes) : "size unavailable"}
                    </span>
                    <Button
                      onClick={() => window.open(resolveUrl(exp.output_url!), "_blank")}
                      icon={<Download size={14} />}
                    >
                      Download video
                    </Button>
                  </div>
                </>
              ) : (
                <Callout tone={exp.status === "failed" ? "danger" : "neutral"}>
                  {exp.status === "failed" ? "The export failed." : "Not ready yet."}
                </Callout>
              )}
            </div>
          </Card>

          <Card>
            <CardHeader title="Subtitles" sub={`${subtitled} of ${segs.length} segments have a translation`} />
            <div className="p-5 flex flex-col gap-3">
              <p className="text-[12px]" style={{ color: "var(--text-muted)" }}>
                Generated here from each segment&apos;s real start and end times and its saved translation.
              </p>
              <div className="flex gap-2 flex-wrap">
                <Button
                  variant="secondary"
                  disabled={subtitled === 0}
                  icon={<Subtitles size={14} />}
                  onClick={() => {
                    downloadText("subtitles.srt", buildSrt(segs), "text/plain;charset=utf-8");
                    toast("success", "subtitles.srt downloaded.");
                  }}
                >
                  SRT
                </Button>
                <Button
                  variant="secondary"
                  disabled={subtitled === 0}
                  icon={<FileText size={14} />}
                  onClick={() => {
                    downloadText("subtitles.vtt", buildVtt(segs), "text/vtt;charset=utf-8");
                    toast("success", "subtitles.vtt downloaded.");
                  }}
                >
                  VTT
                </Button>
              </div>
              {subtitled === 0 && (
                <span className="text-[12px]" style={{ color: "var(--text-dim)" }}>
                  No translated lines yet, so there is nothing to write.
                </span>
              )}
              <Callout tone="neutral" title="Not available">
                Audio-only and per-language files need endpoints this backend does not expose: it muxes one artefact
                per project.
              </Callout>
            </div>
          </Card>
        </div>
      )}

      {exp?.qa_report?.overall && (
        <Card>
          <CardHeader title="QA summary" />
          <dl className="grid grid-cols-2 md:grid-cols-4 gap-4 p-5">
            {Object.entries(exp.qa_report.overall).map(([k, v]) => (
              <div key={k}>
                <dt className="text-[11px]" style={{ color: "var(--text-muted)" }}>{k.replace(/_/g, " ")}</dt>
                <dd className="text-[15px] font-semibold mt-0.5" style={{ color: "var(--text)" }}>{String(v)}</dd>
              </div>
            ))}
          </dl>
        </Card>
      )}

      {qaSegs.length > 0 && (
        <Card className="overflow-hidden">
          <CardHeader title="Per-segment QA" sub="From the mux's own report" />
          <div className="overflow-x-auto">
            <table className="w-full text-left" style={{ minWidth: 760 }}>
              <caption className="sr-only">Per-segment quality report</caption>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  {["Starts", "Speech", "Emotion", "Tempo", "Plays", "Overrun", "Sync offset"].map((h) => (
                    <th key={h} scope="col" className="px-4 py-2.5 text-[11px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-dim)" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {qaSegs.map((r) => {
                  const over = r.overrun_ms ?? 0;
                  return (
                    <tr key={r.segment_id} style={{ borderBottom: "1px solid var(--border)" }}>
                      <td className="px-4 py-2.5 text-[12px]" style={{ color: "var(--text-mid)", fontFamily: "JetBrains Mono, monospace" }}>{msToTimecode(r.start_ms)}</td>
                      <td className="px-4 py-2.5 text-[12px]" style={{ color: "var(--text-muted)" }}>{r.has_speech ? "yes" : "silent"}</td>
                      <td className="px-4 py-2.5 text-[12px]" style={{ color: "var(--text-muted)" }}>{r.emotion_label ?? "—"}</td>
                      <td className="px-4 py-2.5 text-[12px]" style={{ color: (r.tempo ?? 1) > 1.15 ? "var(--warning)" : "var(--text-muted)" }}>
                        {r.tempo != null ? `×${r.tempo.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-4 py-2.5 text-[12px]" style={{ color: "var(--text-muted)" }}>{r.fitted_ms != null ? `${r.fitted_ms} ms` : "—"}</td>
                      <td className="px-4 py-2.5 text-[12px]" style={{ color: over > 0 ? "var(--danger)" : "var(--success)" }}>
                        {r.overrun_ms != null ? `${r.overrun_ms} ms` : "—"}
                      </td>
                      <td className="px-4 py-2.5 text-[12px]" style={{ color: "var(--text-muted)" }}>{r.sync_offset_pct != null ? `${r.sync_offset_pct}%` : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
