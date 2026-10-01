/**
 * Waveform timeline over the dubbed track, with the project's segments drawn
 * on it as regions and a playhead that stays in step with the <audio>/<video>
 * element the editor is already using.
 *
 * wavesurfer.js draws the waveform of whatever audio URL it is given. If
 * there is no dubbed export yet there is nothing to draw, and this renders a
 * plain segment ruler instead of a fake waveform -- a drawn-from-nothing
 * waveform is exactly the kind of invented data this project forbids.
 */
import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

import type { SegmentRead } from "@/lib/types";
import { Skeleton, msToTimecode } from "@/ui";

export function Waveform({
  audioUrl,
  segments,
  durationMs,
  currentMs,
  selectedId,
  onSeek,
  onSelect,
  zoom,
  colorFor,
}: {
  /** The muxed output, or null when the project has not produced one yet. */
  audioUrl: string | null;
  segments: SegmentRead[];
  durationMs: number;
  currentMs: number;
  selectedId: string | null;
  onSeek: (ms: number) => void;
  onSelect: (id: string) => void;
  /** 1 = whole timeline fits; higher scrolls horizontally. */
  zoom: number;
  colorFor: (s: SegmentRead) => string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const ws = useRef<WaveSurfer | null>(null);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!host.current || !audioUrl) return;
    setReady(false);
    setFailed(false);

    const style = getComputedStyle(document.documentElement);
    const instance = WaveSurfer.create({
      container: host.current,
      height: 64,
      waveColor: style.getPropertyValue("--text-dim").trim() || "#5d6577",
      progressColor: style.getPropertyValue("--accent").trim() || "#8b5cf6",
      cursorColor: "transparent", // the playhead below is ours, so there is one
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      interact: true,
      url: audioUrl,
    });
    ws.current = instance;

    instance.on("ready", () => setReady(true));
    instance.on("error", () => setFailed(true));
    instance.on("interaction", () => {
      onSeek(Math.round(instance.getCurrentTime() * 1000));
    });

    return () => {
      ws.current = null;
      instance.destroy();
    };
    // onSeek is intentionally not a dep: re-creating the waveform on every
    // parent render would re-download the audio.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioUrl]);

  const width = `${zoom * 100}%`;
  const pct = (ms: number) => (durationMs > 0 ? (ms / durationMs) * 100 : 0);

  return (
    <div className="rounded-xl overflow-hidden" style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
      <div className="overflow-x-auto">
        <div style={{ width, minWidth: "100%", position: "relative" }}>
          {/* Waveform, when there is audio to draw. */}
          {audioUrl && !failed && (
            <div className="px-0 pt-3" style={{ opacity: ready ? 1 : 0.35 }}>
              <div ref={host} />
            </div>
          )}
          {audioUrl && !ready && !failed && <Skeleton className="h-16 mx-3 mt-3" />}
          {(!audioUrl || failed) && (
            <div className="h-16 mx-3 mt-3 rounded-lg flex items-center justify-center text-[12px]"
              style={{ background: "var(--surface-hover)", color: "var(--text-dim)" }}>
              {failed ? "The dubbed track could not be decoded for display." : "No dubbed track yet — segments only."}
            </div>
          )}

          {/* Segment regions. Clicking one selects it; they are real rows. */}
          <div className="relative h-9 mx-3 my-3 rounded-md" style={{ background: "var(--surface-hover)" }}>
            {segments.map((s) => {
              const left = pct(s.start_ms);
              const w = Math.max(pct(s.end_ms - s.start_ms), 0.3);
              const selected = s.id === selectedId;
              return (
                <button
                  key={s.id}
                  onClick={() => onSelect(s.id)}
                  title={`#${s.index} ${msToTimecode(s.start_ms)} — ${(s.translated_text || s.source_text || "").slice(0, 80)}`}
                  aria-label={`Segment ${s.index} at ${msToTimecode(s.start_ms)}`}
                  className="absolute top-1 bottom-1 rounded-sm transition-all"
                  style={{
                    left: `${left}%`,
                    width: `${w}%`,
                    background: colorFor(s),
                    opacity: selected ? 1 : 0.6,
                    outline: selected ? "2px solid var(--text)" : "none",
                    outlineOffset: 1,
                  }}
                />
              );
            })}

            {/* Playhead */}
            <div
              className="absolute top-0 bottom-0 w-0.5 pointer-events-none"
              style={{ left: `${pct(currentMs)}%`, background: "var(--text)" }}
              aria-hidden="true"
            />
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between px-3 pb-2 text-[11px]" style={{ color: "var(--text-dim)" }}>
        <span>{msToTimecode(currentMs)}</span>
        <span>{msToTimecode(durationMs)}</span>
      </div>
    </div>
  );
}
