/**
 * The seven pipeline stages as a horizontal stepper, driven by the live
 * WebSocket events (useProjectEvents) seeded from the project row
 * (runState.seedStages) so a page opened mid-run is not blank.
 *
 * Stage keys and order come from PIPELINE_STAGE_ORDER in lib/types.ts, which
 * mirrors app/pipeline/tasks.py. Nothing here invents a stage.
 */
import {
  AudioLines,
  Check,
  FileAudio,
  Film,
  Languages,
  Loader2,
  Mic,
  Scissors,
  SmilePlus,
  X,
} from "lucide-react";
import { motion } from "framer-motion";

import { PIPELINE_STAGE_LABELS, type PipelineStageKey } from "@/lib/types";
import type { StageState } from "@/lib/useProjectEvents";
import { useMediaQuery } from "@/ui";

const STAGE_ICON: Record<PipelineStageKey, typeof Mic> = {
  extract_audio: FileAudio,
  chunk_and_diarize: Scissors,
  transcribe: Mic,
  detect_emotion: SmilePlus,
  translate: Languages,
  synthesize: AudioLines,
  mux_export: Film,
};

function stageSeconds(s: StageState): string | null {
  if (s.startedAt == null) return null;
  const end = s.done && s.completedAt != null ? s.completedAt : Date.now();
  const secs = Math.round((end - s.startedAt) / 1000);
  if (secs < 1) return null;
  return secs < 60 ? `${secs}s` : `${Math.floor(secs / 60)}m ${secs % 60}s`;
}

export function StageTracker({ stages, failedStage }: { stages: StageState[]; failedStage?: string | null }) {
  const isNarrow = useMediaQuery("(max-width: 1023px)");

  return (
    <ol className={`flex ${isNarrow ? "flex-col gap-3" : "items-start gap-1"}`} aria-label="Pipeline stages">
      {stages.map((s, i) => {
        const Icon = STAGE_ICON[s.key as PipelineStageKey] ?? Mic;
        const failed = failedStage === s.key;
        const label = PIPELINE_STAGE_LABELS[s.key as PipelineStageKey] ?? s.key;
        const count = s.completed != null && s.total != null ? `${s.completed}/${s.total}` : null;
        const took = stageSeconds(s);

        const ring = failed
          ? "var(--danger)"
          : s.done
            ? "var(--success)"
            : s.active
              ? "var(--running)"
              : "var(--border-strong)";
        const fg = failed
          ? "var(--danger)"
          : s.done
            ? "var(--success)"
            : s.active
              ? "var(--running)"
              : "var(--text-dim)";

        return (
          <li key={s.key} className={isNarrow ? "flex items-center gap-3" : "flex-1 flex flex-col items-center gap-2 min-w-0"}>
            <div className={isNarrow ? "flex items-center gap-3 flex-shrink-0" : "flex items-center w-full"}>
              {/* connector before */}
              {!isNarrow && i > 0 && (
                <div className="h-0.5 flex-1" style={{ background: stages[i - 1].done ? "var(--success)" : "var(--border)" }} aria-hidden="true" />
              )}
              <motion.div
                className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0"
                style={{ border: `1.5px solid ${ring}`, background: s.done || failed ? "var(--bg-elevated)" : "var(--surface)", color: fg }}
                animate={s.active && !failed ? { scale: [1, 1.06, 1] } : { scale: 1 }}
                transition={s.active ? { duration: 1.6, repeat: Infinity, ease: "easeInOut" } : { duration: 0.2 }}
                aria-hidden="true"
              >
                {failed ? <X size={15} /> : s.done ? <Check size={15} /> : s.active ? <Loader2 size={15} className="animate-spin" /> : <Icon size={15} />}
              </motion.div>
              {/* connector after */}
              {!isNarrow && i < stages.length - 1 && (
                <div className="h-0.5 flex-1" style={{ background: s.done ? "var(--success)" : "var(--border)" }} aria-hidden="true" />
              )}
            </div>

            <div className={isNarrow ? "flex-1 min-w-0" : "flex flex-col items-center text-center min-w-0 w-full"}>
              <span className="text-[12px] font-medium truncate max-w-full" style={{ color: s.active || s.done ? "var(--text)" : "var(--text-dim)" }}>
                {label}
              </span>
              <span className="text-[11px] truncate max-w-full" style={{ color: "var(--text-dim)" }}>
                {failed
                  ? "failed"
                  : s.done
                    ? took ?? "done"
                    : s.active
                      ? `${Math.round(s.progress * 100)}%${count ? ` · ${count}` : ""}${s.eta ? ` · ~${s.eta}` : ""}`
                      : "pending"}
              </span>
              {isNarrow && (s.active || s.done) && (
                <div className="h-0.5 rounded-full overflow-hidden mt-1" style={{ background: "var(--surface-hover)" }}>
                  <div className="h-full rounded-full" style={{ width: `${s.done ? 100 : Math.round(s.progress * 100)}%`, background: fg }} />
                </div>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
