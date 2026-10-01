/**
 * Drag-and-drop upload with the validation the backend actually publishes:
 * accepted_formats and max_upload_mb from /api/capabilities, never a list of
 * our own. Shows a real <video> preview and its real duration, read from the
 * browser's own metadata rather than estimated.
 */
import { FileVideo, UploadCloud, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button, ProgressBar, fmtBytes, msToTimecode } from "@/ui";

export interface PickedFile {
  file: File;
  durationMs: number | null;
  previewUrl: string;
}

export function UploadZone({
  accept,
  maxMb,
  picked,
  onPick,
  onClear,
  progress,
  disabled,
}: {
  accept: string[];
  maxMb: number;
  picked: PickedFile | null;
  onPick: (p: PickedFile) => void;
  onClear: () => void;
  /** 0..1 while the PUT is in flight, null when idle. */
  progress: number | null;
  disabled?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validate = useCallback(
    (f: File): string | null => {
      const maxBytes = maxMb * 1024 * 1024;
      if (f.size > maxBytes) {
        return `That file is ${fmtBytes(f.size)}; this deployment accepts up to ${maxMb} MB.`;
      }
      // f.type can be empty for some containers; only reject a type we were
      // actually told about, so a valid file with no reported MIME still goes.
      if (accept.length && f.type && !accept.includes(f.type)) {
        return `${f.type} is not accepted here. Allowed: ${accept.join(", ")}.`;
      }
      return null;
    },
    [accept, maxMb],
  );

  const take = useCallback(
    (f: File | null) => {
      setError(null);
      if (!f) return;
      const bad = validate(f);
      if (bad) {
        setError(bad);
        return;
      }
      const url = URL.createObjectURL(f);
      // Read the real duration from the file itself. An estimate from bytes
      // would be a guess shown as a fact.
      const probe = document.createElement("video");
      probe.preload = "metadata";
      probe.src = url;
      const done = (ms: number | null) => onPick({ file: f, durationMs: ms, previewUrl: url });
      probe.onloadedmetadata = () => done(Number.isFinite(probe.duration) ? Math.round(probe.duration * 1000) : null);
      probe.onerror = () => done(null);
    },
    [onPick, validate],
  );

  useEffect(() => {
    return () => {
      if (picked) URL.revokeObjectURL(picked.previewUrl);
    };
  }, [picked]);

  if (picked) {
    return (
      <div className="flex flex-col gap-3">
        <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--border)", background: "var(--bg)" }}>
          <video
            src={picked.previewUrl}
            controls
            className="w-full"
            style={{ maxHeight: 280, background: "#000" }}
            aria-label={`Preview of ${picked.file.name}`}
          />
          <div className="flex items-center gap-3 px-4 py-3" style={{ borderTop: "1px solid var(--border)" }}>
            <FileVideo size={16} style={{ color: "var(--accent)", flexShrink: 0 }} aria-hidden="true" />
            <div className="min-w-0 flex-1">
              <div className="text-[13px] font-medium truncate" style={{ color: "var(--text)" }}>{picked.file.name}</div>
              <div className="text-[12px]" style={{ color: "var(--text-muted)" }}>
                {fmtBytes(picked.file.size)}
                {picked.durationMs != null && ` · ${msToTimecode(picked.durationMs)}`}
                {picked.durationMs == null && " · duration unavailable"}
              </div>
            </div>
            {progress == null && (
              <Button variant="ghost" size="sm" icon={<X size={13} />} onClick={onClear} disabled={disabled}>
                Remove
              </Button>
            )}
          </div>
          {progress != null && (
            <div className="px-4 pb-3 flex flex-col gap-1.5">
              <div className="flex justify-between text-[11px]" style={{ color: "var(--text-muted)" }}>
                <span>Uploading…</span>
                <span>{Math.round(progress * 100)}%</span>
              </div>
              <ProgressBar value={progress} label="Upload progress" />
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <input
        ref={input}
        type="file"
        hidden
        accept={accept.join(",")}
        onChange={(e) => take(e.target.files?.[0] ?? null)}
      />
      <button
        type="button"
        disabled={disabled}
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files?.[0] ?? null); }}
        className="rounded-xl flex flex-col items-center justify-center gap-3 transition-all px-6 disabled:opacity-50"
        style={{
          minHeight: 220,
          border: `2px dashed ${over ? "var(--accent)" : "var(--border-strong)"}`,
          background: over ? "var(--accent-soft)" : "var(--bg-elevated)",
        }}
      >
        <div
          className="w-12 h-12 rounded-xl flex items-center justify-center"
          style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
          aria-hidden="true"
        >
          <UploadCloud size={22} />
        </div>
        <div className="text-center">
          <div className="text-[14px] font-medium" style={{ color: "var(--text)" }}>
            Drop a video here, or click to choose
          </div>
          <div className="text-[12px] mt-1" style={{ color: "var(--text-muted)" }}>
            {accept.length ? accept.map((a) => a.replace("video/", "").toUpperCase()).join(" · ") : "video"} · up to {maxMb} MB
          </div>
        </div>
      </button>
      {error && (
        <span role="alert" className="text-[12px]" style={{ color: "var(--danger)" }}>{error}</span>
      )}
    </div>
  );
}
