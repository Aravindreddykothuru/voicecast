/**
 * Studio editor: the dubbed track on a waveform timeline, a segment table
 * synced to it, and per-segment re-translate / re-voice.
 *
 * What the backend does and does not give us, stated rather than papered over:
 *  - The muxed output (ExportRead.output_url) can be played, so the waveform
 *    and the playhead are real.
 *  - There is NO URL for the original video on ProjectRead, so a true
 *    side-by-side original-vs-dubbed player is not possible. Per segment the
 *    original audio IS available (source_audio_url), so "play original" works
 *    line by line. The missing endpoint is listed in the report.
 *  - Duration fit comes from the export's QA report (tempo, fitted_ms,
 *    overrun_ms) — the mux's own numbers. Comparing raw clip length against
 *    the segment end used to flag every normal fit as a failure.
 *
 * Carried over from the old editor because each cost a real bug:
 *  - "Re-voice" sends only ["synthesize"] and keeps the saved text; sending
 *    both stages threw the user's edit away and spoke a fresh machine
 *    translation.
 *  - Unsaved text is never sent: save first, then re-voice.
 *  - Audio is cache-busted on updated_at, because the object key does not
 *    change when a segment is re-voiced and the browser kept the old clip.
 *  - While the project is running, editing is paused and the page polls.
 */
import {
  Download,
  Languages,
  Lock,
  LockOpen,
  Pause,
  Play,
  RefreshCw,
  Save,
  Volume2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Waveform } from "@/components/Waveform";
import {
  getExport,
  getProject,
  listSegments,
  patchSegment,
  regenerateSegment,
  resolveUrl,
  type RegenerateStages,
} from "@/lib/api";
import { emotionDisplay, emotionIndexOf, sourceLanguageName, useReadyCapabilities } from "@/lib/capabilities";
import { emotionColor, UNCERTAIN_COLOR } from "@/lib/theme";
import { projectError, type ProjectRead, type SegmentQaEntry, type SegmentRead } from "@/lib/types";
import {
  Button,
  Callout,
  Card,
  EmptyState,
  Select,
  Skeleton,
  StatusBadge,
  Textarea,
  msToTimecode,
  statusTone,
  useToast,
} from "@/ui";
import type { Screen } from "@/components/AppShell";

/** Green / amber / red from the mux's own overrun, not a guess. */
function fitOf(qa: SegmentQaEntry | undefined): { tone: "success" | "warning" | "danger" | "neutral"; label: string; title: string } {
  if (!qa) return { tone: "neutral", label: "—", title: "No QA report yet: timing is fitted when the export is muxed." };
  const over = qa.overrun_ms ?? 0;
  const tempo = qa.tempo ?? 1;
  if (over > 0) return { tone: "danger", label: `+${Math.round(over)}ms`, title: `Runs ${Math.round(over)} ms past the next line after timing fit.` };
  if (tempo > 1.15) return { tone: "warning", label: `×${tempo.toFixed(2)}`, title: `Fits, but had to be sped up ${tempo.toFixed(2)}× to do it.` };
  return { tone: "success", label: "fits", title: `Fits at ×${tempo.toFixed(2)}.` };
}

export function Editor({ projectId, go }: { projectId: string | null; go: (s: Screen) => void }) {
  const caps = useReadyCapabilities();
  const toast = useToast();

  const [project, setProject] = useState<ProjectRead | null>(null);
  const [segments, setSegments] = useState<SegmentRead[] | null>(null);
  const [qa, setQa] = useState<Map<string, SegmentQaEntry>>(new Map());
  const [dubUrl, setDubUrl] = useState<string | null>(null);
  const [dubBytes, setDubBytes] = useState<number | null>(null);
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  const [mediaTooBig, setMediaTooBig] = useState(false);
  const [selId, setSelId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [emotionDraft, setEmotionDraft] = useState<string>("");
  const [locked, setLocked] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [currentMs, setCurrentMs] = useState(0);

  const video = useRef<HTMLVideoElement | null>(null);
  const segAudio = useRef<HTMLAudioElement | null>(null);

  const load = useCallback(async () => {
    if (!projectId) return;
    try {
      const [p, rows, exp] = await Promise.all([getProject(projectId), listSegments(projectId), getExport(projectId).catch(() => null)]);
      setProject(p);
      setSegments(rows);
      setSelId((cur) => cur ?? rows[0]?.id ?? null);
      setQa(new Map((exp?.qa_report?.segments ?? []).map((r) => [r.segment_id, r])));
      setDubUrl(exp?.output_url ? resolveUrl(exp.output_url) : null);
      setDubBytes(exp?.output_size_bytes ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load the project");
    }
  }, [projectId]);

  useEffect(() => { void load(); }, [load]);

  // One fetch, one blob, both consumers.
  //
  // The <video> and the waveform need the same bytes. Pointed at the same
  // URL they collide: the media element's load is a no-cors request, Chrome
  // reuses that opaque response for wavesurfer's fetch, and the CORS check
  // fails ("Failed to fetch") even though the server sends the right header.
  // Reproduced: the fetch alone succeeds, the same fetch during playback does
  // not. Cache-busting one of them would invalidate an S3 presigned
  // signature, so instead the file is fetched once and both read the blob.
  //
  // That holds the export in memory, which is fine for a dub of this size and
  // not fine for an arbitrarily large one, so it is bounded: past the limit
  // the video streams normally and the waveform is skipped rather than
  // silently eating memory.
  const MAX_INLINE_BYTES = 150 * 1024 * 1024;
  useEffect(() => {
    if (!dubUrl) {
      setMediaUrl(null);
      return;
    }
    if (dubBytes != null && dubBytes > MAX_INLINE_BYTES) {
      setMediaTooBig(true);
      setMediaUrl(dubUrl);
      return;
    }
    let alive = true;
    let objectUrl: string | null = null;
    fetch(dubUrl)
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((blob) => {
        if (!alive) return;
        objectUrl = URL.createObjectURL(blob);
        setMediaUrl(objectUrl);
      })
      .catch(() => {
        // Fall back to streaming the URL directly: the player still works,
        // the waveform is the part that cannot draw.
        if (alive) setMediaUrl(dubUrl);
      });
    return () => {
      alive = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [dubUrl, dubBytes]);

  // A regenerate used to be fire-and-forget: the screen never refreshed, so
  // new audio never appeared and a failure was invisible.
  const running = !!project && (project.status === "queued" || project.status === "processing") && !project.stalled;
  const wasRunning = useRef(false);
  useEffect(() => {
    if (running) {
      wasRunning.current = true;
      const t = setInterval(() => { void load(); }, 3000);
      return () => clearInterval(t);
    }
    if (wasRunning.current && project) {
      wasRunning.current = false;
      if (project.status === "ready") toast("success", "Re-voiced and re-muxed.");
    }
  }, [running, load, project, toast]);

  const segs = segments ?? [];
  const seg = segs.find((s) => s.id === selId) ?? null;
  const selIndex = segs.findIndex((s) => s.id === selId);

  useEffect(() => {
    setDraft(seg?.translated_text ?? "");
    setEmotionDraft(seg?.emotion_label ?? "");
  }, [seg?.id, seg?.translated_text, seg?.emotion_label]);

  const durationMs = useMemo(
    () => segs.reduce((max, s) => Math.max(max, s.end_ms, s.start_ms + (qa.get(s.id)?.fitted_ms ?? s.tts_duration_ms ?? 0)), 1),
    [segs, qa],
  );

  const dirty = !!seg && draft !== (seg.translated_text ?? "");
  const isLocked = !!seg && locked.has(seg.id);

  const colorFor = useCallback(
    (s: SegmentRead) => {
      const idx = emotionIndexOf(caps, s.emotion_label);
      if (idx == null) return "var(--accent)";
      return (s.emotion_score ?? 0) < caps.emotion_confidence_floor ? UNCERTAIN_COLOR : emotionColor(idx);
    },
    [caps],
  );

  // ── actions ─────────────────────────────────────────────────────────────
  const save = useCallback(async () => {
    if (!seg || isLocked) return;
    setBusy(true);
    setError(null);
    try {
      const patch: { translated_text?: string; emotion_label?: string } = {};
      if (draft !== (seg.translated_text ?? "")) patch.translated_text = draft;
      if (emotionDraft && emotionDraft !== (seg.emotion_label ?? "")) patch.emotion_label = emotionDraft;
      if (Object.keys(patch).length === 0) return;
      await patchSegment(seg.id, patch);
      await load();
      toast("success", "Saved. Re-voice the line to hear it in the dub.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setBusy(false);
    }
  }, [seg, draft, emotionDraft, isLocked, load, toast]);

  const regen = useCallback(
    async (stages: RegenerateStages) => {
      if (!seg || isLocked) return;
      setBusy(true);
      setError(null);
      try {
        await regenerateSegment(seg.id, stages);
        toast("running", "Regenerating — the export is re-muxed when it finishes.");
        await load();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Regenerate failed");
      } finally {
        setBusy(false);
      }
    },
    [seg, isLocked, load, toast],
  );

  const select = useCallback(
    (id: string) => {
      setSelId(id);
      const s = segs.find((x) => x.id === id);
      if (s) {
        setCurrentMs(s.start_ms);
        if (video.current) video.current.currentTime = s.start_ms / 1000;
      }
    },
    [segs],
  );

  const step = useCallback(
    (delta: number) => {
      if (segs.length === 0) return;
      const next = Math.max(0, Math.min(segs.length - 1, (selIndex < 0 ? 0 : selIndex) + delta));
      select(segs[next].id);
    },
    [segs, selIndex, select],
  );

  const togglePlay = useCallback(() => {
    const v = video.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  }, []);

  const playSegment = useCallback((s: SegmentRead, which: "original" | "dubbed") => {
    const url = which === "original" ? s.source_audio_url : s.tts_audio_url;
    if (!url) return;
    const a = segAudio.current;
    if (!a) return;
    a.src = `${resolveUrl(url)}?v=${encodeURIComponent(s.updated_at)}`;
    void a.play();
  }, []);

  // ── keyboard shortcuts ──────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        void save();
        return;
      }
      if (typing) return;
      if (e.code === "Space") { e.preventDefault(); togglePlay(); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [save, togglePlay, step]);

  if (!projectId) {
    return (
      <Card className="m-6">
        <EmptyState title="No project open" body="Open one from the dashboard." action={<Button onClick={() => go("dashboard")}>Go to dashboard</Button>} />
      </Card>
    );
  }

  const perr = project ? projectError(project) : null;
  const changedCount = segs.filter((s) => s.status === "translated" || s.emotion_overridden).length;

  return (
    <div className="page-container gap-6">
      <audio ref={segAudio} hidden />

      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <h1 className="text-[30px] font-semibold truncate" style={{ color: "var(--text)" }}>
            {project?.title ?? "Studio"}
          </h1>
          <p className="text-[14px] mt-1" style={{ color: "var(--text-muted)" }}>
            {segments ? `${segs.length} segments` : "Loading…"}
            {changedCount > 0 && ` · ${changedCount} edited`}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[12px] hidden lg:inline" style={{ color: "var(--text-dim)" }}>
            Space play · ←/→ segment · Ctrl+S save
          </span>
          <Button variant="ghost" onClick={() => go("export")} icon={<Download size={14} />}>Export</Button>
          <Button
            variant="secondary"
            onClick={() => regen(["synthesize"])}
            disabled={busy || running || !seg}
            icon={<RefreshCw size={14} />}
            title="Re-renders only the selected segment, then re-muxes"
          >
            Re-render segment
          </Button>
        </div>
      </div>

      {error && <Callout tone="danger">{error}</Callout>}
      {perr && <Callout tone="danger" title="The last run failed">{perr.message}</Callout>}
      {project?.stalled && <Callout tone="warning" title="No progress">{project.stalled_reason}</Callout>}
      {running && <Callout tone="running" title="Processing">Editing is paused while the project is running.</Callout>}

      {/* Player + waveform */}
      <Card className="p-4 flex flex-col gap-3">
        <div className="grid gap-4" style={{ gridTemplateColumns: "minmax(0,1fr)" }}>
          {mediaUrl ? (
            <div className="flex flex-col sm:flex-row gap-4">
              <video
                ref={video}
                src={mediaUrl}
                controls
                className="rounded-lg w-full sm:w-1/2"
                style={{ background: "#000", maxHeight: 280 }}
                onTimeUpdate={(e) => setCurrentMs(Math.round(e.currentTarget.currentTime * 1000))}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                aria-label="Dubbed output"
              />
              <div className="flex-1 flex flex-col gap-2 justify-center">
                <div className="text-[13px] font-medium" style={{ color: "var(--text-mid)" }}>Dubbed output</div>
                <p className="text-[13px]" style={{ color: "var(--text-dim)" }}>
                  The original video has no URL on this API, so it cannot be shown beside this one. Per line, the
                  original audio plays from the segment table.
                </p>
                <div className="flex gap-2">
                  <Button size="sm" variant="secondary" onClick={togglePlay} icon={playing ? <Pause size={13} /> : <Play size={13} />}>
                    {playing ? "Pause" : "Play"}
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setZoom((z) => Math.max(1, z - 1))} icon={<ZoomOut size={13} />} aria-label="Zoom out" />
                  <Button size="sm" variant="ghost" onClick={() => setZoom((z) => Math.min(8, z + 1))} icon={<ZoomIn size={13} />} aria-label="Zoom in" />
                </div>
              </div>
            </div>
          ) : (
            <Callout tone="neutral" title="No dubbed output yet">
              The waveform draws the muxed export. Until the pipeline produces one, the timeline below shows segments
              only.
            </Callout>
          )}
        </div>

        {segments ? (
          <Waveform
            audioUrl={mediaTooBig ? null : mediaUrl}
            segments={segs}
            durationMs={durationMs}
            currentMs={currentMs}
            selectedId={selId}
            onSeek={(ms) => { setCurrentMs(ms); if (video.current) video.current.currentTime = ms / 1000; }}
            onSelect={select}
            zoom={zoom}
            colorFor={colorFor}
          />
        ) : (
          <Skeleton className="h-28" />
        )}
      </Card>

      {/* Segment table */}
      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left" style={{ minWidth: 900 }}>
            <caption className="sr-only">Segments, synced to the timeline</caption>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["#", "Time", "Speaker", "Original", "Translation", "Emotion", "Fit", ""].map((h) => (
                  <th key={h} scope="col" className="px-3 py-2.5 text-[12px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-dim)" }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {segments === null &&
                Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i}><td colSpan={8} className="px-3 py-2"><Skeleton className="h-6" /></td></tr>
                ))}

              {segs.map((s) => {
                const active = s.id === selId;
                const fit = fitOf(qa.get(s.id));
                const st = statusTone(s.status);
                const lockedRow = locked.has(s.id);
                return (
                  <tr
                    key={s.id}
                    onClick={() => select(s.id)}
                    className="cursor-pointer transition-colors"
                    style={{
                      borderBottom: "1px solid var(--border)",
                      background: active ? "var(--accent-soft)" : "transparent",
                    }}
                  >
                    <td className="px-3 py-2.5 text-[13px]" style={{ color: "var(--text-dim)" }}>{s.index}</td>
                    <td className="px-3 py-2.5 text-[13px] whitespace-nowrap" style={{ color: "var(--text-mid)", fontFamily: "JetBrains Mono, monospace" }}>
                      {msToTimecode(s.start_ms)}
                    </td>
                    <td className="px-3 py-2.5 text-[13px]" style={{ color: "var(--text-muted)" }}>{s.speaker_id ?? "—"}</td>
                    <td className="px-3 py-2.5 text-[13px] max-w-[240px]" style={{ color: "var(--text-muted)" }}>
                      <span className="line-clamp-2">{s.source_text ?? "—"}</span>
                    </td>
                    <td className="px-3 py-2.5 text-[13px] max-w-[280px]" style={{ color: "var(--text)" }}>
                      <span className="line-clamp-2">{s.translated_text ?? "—"}</span>
                    </td>
                    <td className="px-3 py-2.5">
                      <span
                        className="inline-flex items-center gap-1.5 text-[12px] px-2 py-0.5 rounded-full"
                        style={{ background: "var(--surface-hover)", color: "var(--text-mid)" }}
                      >
                        <span className="w-2 h-2 rounded-full" style={{ background: colorFor(s) }} aria-hidden="true" />
                        {emotionDisplay(caps, s.emotion_label, s.emotion_score).text}
                      </span>
                    </td>
                    <td className="px-3 py-2.5" title={fit.title}>
                      <StatusBadge tone={fit.tone}>{fit.label}</StatusBadge>
                    </td>
                    <td className="px-3 py-2.5">
                      <div className="flex items-center gap-1 justify-end">
                        {s.source_audio_url && (
                          <button
                            onClick={(e) => { e.stopPropagation(); playSegment(s, "original"); }}
                            aria-label={`Play original audio for segment ${s.index}`}
                            title="Play original"
                            className="w-7 h-7 rounded-md flex items-center justify-center"
                            style={{ color: "var(--text-muted)" }}
                          >
                            <Volume2 size={13} />
                          </button>
                        )}
                        {s.tts_audio_url && (
                          <button
                            onClick={(e) => { e.stopPropagation(); playSegment(s, "dubbed"); }}
                            aria-label={`Play dubbed audio for segment ${s.index}`}
                            title="Play dubbed"
                            className="w-7 h-7 rounded-md flex items-center justify-center"
                            style={{ color: "var(--accent)" }}
                          >
                            <Play size={13} />
                          </button>
                        )}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setLocked((l) => {
                              const n = new Set(l);
                              if (n.has(s.id)) n.delete(s.id); else n.add(s.id);
                              return n;
                            });
                          }}
                          aria-label={lockedRow ? `Unlock segment ${s.index}` : `Lock segment ${s.index}`}
                          title={lockedRow ? "Unlock" : "Lock against edits"}
                          className="w-7 h-7 rounded-md flex items-center justify-center"
                          style={{ color: lockedRow ? "var(--warning)" : "var(--text-dim)" }}
                        >
                          {lockedRow ? <Lock size={13} /> : <LockOpen size={13} />}
                        </button>
                        <StatusBadge tone={st.tone}>{st.label}</StatusBadge>
                      </div>
                    </td>
                  </tr>
                );
              })}

              {segments && segs.length === 0 && (
                <tr><td colSpan={8}><EmptyState title="No segments yet" body="They appear after chunking and transcription." /></td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Selected segment editor */}
      {seg && (
        <Card className="p-5 flex flex-col gap-4">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div className="flex items-center gap-3 text-[13px]" style={{ color: "var(--text-muted)" }}>
              <span style={{ color: "var(--text)" }}>Segment {seg.index}</span>
              <span style={{ fontFamily: "JetBrains Mono, monospace" }}>{msToTimecode(seg.start_ms)}–{msToTimecode(seg.end_ms)}</span>
              {seg.detected_language && <span>{sourceLanguageName(caps, seg.detected_language)}</span>}
              {seg.emotion_overridden && <StatusBadge tone="warning">emotion overridden</StatusBadge>}
              {isLocked && <StatusBadge tone="warning">locked</StatusBadge>}
            </div>
            {seg.tts_audio_url && (
              <audio
                controls
                key={seg.updated_at}
                src={`${resolveUrl(seg.tts_audio_url)}?v=${encodeURIComponent(seg.updated_at)}`}
                className="h-8"
                aria-label={`Dubbed audio for segment ${seg.index}`}
              />
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div>
              <div className="text-[12px] uppercase tracking-wider mb-1.5" style={{ color: "var(--text-dim)" }}>Original</div>
              <p className="text-[14px] leading-relaxed" style={{ color: "var(--text-muted)" }}>{seg.source_text ?? "—"}</p>
            </div>
            <div className="flex flex-col gap-3">
              <div>
                <label htmlFor="translation" className="block text-[12px] uppercase tracking-wider mb-1.5" style={{ color: "var(--accent)" }}>
                  Translation
                </label>
                <Textarea
                  id="translation"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  rows={3}
                  disabled={isLocked || running}
                />
              </div>
              <div className="flex items-end gap-3">
                <div className="flex-1">
                  <label htmlFor="emotion" className="block text-[12px] uppercase tracking-wider mb-1.5" style={{ color: "var(--text-dim)" }}>
                    Emotion
                  </label>
                  <Select
                    id="emotion"
                    value={emotionDraft}
                    onChange={(e) => setEmotionDraft(e.target.value)}
                    disabled={isLocked || running}
                  >
                    <option value="">{seg.emotion_label ?? "not detected"}</option>
                    {caps.emotions.map((em) => (
                      <option key={em.label} value={em.label}>{em.label}</option>
                    ))}
                  </Select>
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            <Button onClick={save} loading={busy} disabled={running || isLocked || (!dirty && (!emotionDraft || emotionDraft === (seg.emotion_label ?? "")))} icon={<Save size={14} />}>
              Save
            </Button>
            <Button
              variant="secondary"
              onClick={() => regen(["synthesize"])}
              disabled={busy || running || isLocked || dirty || !(seg.translated_text ?? "").trim()}
              icon={<Volume2 size={14} />}
            >
              Re-voice this line
            </Button>
            <Button
              variant="ghost"
              onClick={() => regen(["translate", "synthesize"])}
              disabled={busy || running || isLocked || dirty || !(seg.source_text ?? "").trim()}
              icon={<Languages size={14} />}
            >
              Re-translate &amp; re-voice
            </Button>
            {dirty && (
              <span className="text-[13px]" style={{ color: "var(--warning)" }}>
                Save before re-voicing — unsaved edits are not sent.
              </span>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}
