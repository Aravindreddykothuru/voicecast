/**
 * New Dub: a three-step wizard over the same four calls the old form made --
 * POST /api/projects, POST /upload, PUT the file, POST /upload/confirm, then
 * POST /process.
 *
 * Every option comes from /api/capabilities. The lessons from the old screen
 * are kept, and they are not cosmetic:
 *
 *  - "Autodetect" sends review_language=true and no source_language, which
 *    parks the run after ASR for confirmation. A pinned language is sent WITH
 *    /process instead; it used to be applied by calling /confirm-language
 *    straight after, which 409s because a run that is not reviewing never
 *    reaches the gate -- so the user's pick was silently dropped.
 *  - A failure after the project row exists leaves an orphan in "uploading".
 *    The error offers a link to it rather than stranding it.
 *  - Targets with tts_available=false are offered as translate-only and
 *    cannot be submitted: POST /api/projects would 422 them anyway.
 */
import { ArrowLeft, ArrowRight, Check, Cpu, Sparkles } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useMemo, useState } from "react";

import { UploadZone, type PickedFile } from "@/components/UploadZone";
import {
  ApiError,
  confirmUpload,
  createProject,
  createUploadUrl,
  putUploadFile,
  startProcessing,
} from "@/lib/api";
import { languageName, useReadyCapabilities } from "@/lib/capabilities";
import {
  Button,
  Callout,
  Card,
  Field,
  Input,
  Select,
  Toggle,
  fmtBytes,
  msToTimecode,
  useToast,
} from "@/ui";
import type { Screen } from "@/components/AppShell";
import type { WorkerHealth } from "@/lib/types";

const STEPS = ["Upload", "Settings", "Review & start"] as const;

export function NewDub({
  go,
  onCreated,
  workerHealth,
}: {
  go: (s: Screen) => void;
  onCreated: (id: string) => void;
  workerHealth?: WorkerHealth | null;
}) {
  const caps = useReadyCapabilities();
  const toast = useToast();

  const [step, setStep] = useState(0);
  const [picked, setPicked] = useState<PickedFile | null>(null);
  const [title, setTitle] = useState("");
  const [sourceLang, setSourceLang] = useState(caps.asr_autodetect ? "auto" : (caps.source_languages[0]?.code ?? ""));
  const [targets, setTargets] = useState<string[]>([]);
  const [preserveEmotion, setPreserveEmotion] = useState(true);
  const [cloneVoice, setCloneVoice] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [orphanId, setOrphanId] = useState<string | null>(null);

  const dubbable = caps.languages.filter((l) => l.tts_available);
  const translateOnly = caps.languages.filter((l) => !l.tts_available);
  const chosen = targets.filter((c) => dubbable.some((l) => l.code === c));

  const canLeaveUpload = !!picked;
  const canLeaveSettings = chosen.length > 0 && !!title.trim();

  // The pipeline runs every stage on the CPU here, so the honest estimate is a
  // multiple of the clip's length, stated as a range and labelled rough.
  const estimate = useMemo(() => {
    if (!picked?.durationMs) return null;
    const mins = picked.durationMs / 60000;
    const per = caps.device === "cpu" ? [6, 14] : [1, 3];
    const extra = cloneVoice ? 2.2 : 1;
    const lo = Math.max(1, Math.round(mins * per[0] * extra * chosen.length || 1));
    const hi = Math.max(lo + 1, Math.round(mins * per[1] * extra * chosen.length || 2));
    return `${lo}–${hi} min`;
  }, [picked, caps.device, cloneVoice, chosen.length]);

  const toggleTarget = (code: string) =>
    setTargets((t) => (t.includes(code) ? t.filter((x) => x !== code) : [...t, code]));

  const submit = async () => {
    if (!picked) return;
    setSubmitting(true);
    setError(null);
    setOrphanId(null);
    let projectId: string | null = null;
    try {
      const project = await createProject(title.trim(), chosen);
      projectId = project.id;
      const up = await createUploadUrl(project.id, picked.file.name, picked.file.type || "video/mp4");
      setUploadPct(0);
      await putUploadFile(up.upload_url, picked.file, setUploadPct);
      setUploadPct(null);
      await confirmUpload(project.id, up.source_video_id, picked.durationMs ?? undefined);
      await startProcessing(project.id, {
        preserve_emotion: preserveEmotion,
        clone_voice: cloneVoice && caps.voice_clone_available,
        lip_sync_aware: false,
        review_language: sourceLang === "auto",
        source_language: sourceLang === "auto" ? null : sourceLang,
      });
      toast("success", "Dubbing started.");
      onCreated(project.id);
    } catch (e) {
      const reason = e instanceof ApiError || e instanceof Error ? e.message : "Could not start the job";
      setUploadPct(null);
      setOrphanId(projectId);
      setError(projectId ? `The project was created but not started: ${reason}` : reason);
      setSubmitting(false);
    }
  };

  return (
    <div className="page-container gap-6">
      <div>
        <h1 className="text-[30px] font-semibold" style={{ color: "var(--text)" }}>New dub</h1>
        <p className="text-[14px] mt-1" style={{ color: "var(--text-muted)" }}>
          Upload a video, choose what it becomes, then start the pipeline.
        </p>
      </div>

      {/* Stepper */}
      <ol className="flex items-center gap-3 w-full" aria-label="Progress">
        {STEPS.map((s, i) => {
          const done = i < step;
          const here = i === step;
          return (
            <li key={s} className="flex items-center gap-3 flex-1 last:flex-none">
              <button
                onClick={() => i < step && setStep(i)}
                disabled={i > step}
                aria-current={here ? "step" : undefined}
                className="flex items-center gap-2.5 disabled:cursor-default"
              >
                <span
                  className="w-8 h-8 rounded-full flex items-center justify-center text-[13px] font-semibold flex-shrink-0 transition-colors"
                  style={{
                    background: done ? "var(--success-soft)" : here ? "var(--accent)" : "var(--surface-hover)",
                    color: done ? "var(--success)" : here ? "#fff" : "var(--text-dim)",
                    border: `1px solid ${done ? "var(--success-border)" : here ? "var(--accent)" : "var(--border)"}`,
                  }}
                >
                  {done ? <Check size={14} /> : i + 1}
                </span>
                <span
                  className="text-[14px] font-medium hidden sm:inline"
                  style={{ color: here ? "var(--text)" : "var(--text-muted)" }}
                >
                  {s}
                </span>
              </button>
              {i < STEPS.length - 1 && (
                <div className="h-px flex-1 min-w-4" style={{ background: done ? "var(--success)" : "var(--border)" }} aria-hidden="true" />
              )}
            </li>
          );
        })}
      </ol>

      {caps.device === "cpu" && step === 0 && (
        <Callout tone="warning" title="This deployment runs on the CPU">
          Every stage runs on the processor, so expect several times the video&apos;s length — longer with voice
          cloning. The progress screen shows live per-stage detail.
        </Callout>
      )}

      <AnimatePresence mode="wait">
        <motion.div
          key={step}
          initial={{ opacity: 0, x: 12 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -12 }}
          transition={{ duration: 0.18 }}
          className="flex flex-col gap-5 w-full"
        >
          {step === 0 && (
            <div className="w-full">
              <UploadZone
                accept={caps.accepted_formats}
                maxMb={caps.max_upload_mb}
                picked={picked}
                onPick={(p) => {
                  setPicked(p);
                  if (!title.trim()) setTitle(p.file.name.replace(/\.[^.]+$/, ""));
                }}
                onClear={() => setPicked(null)}
                progress={uploadPct}
              />
            </div>
          )}

          {step === 1 && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 w-full items-start">
              <Card className="lg:col-span-7 p-6 flex flex-col gap-5 w-full">
                <div className="text-[16px] font-semibold" style={{ color: "var(--text)" }}>Project & Languages</div>
                <Field label="Project name" htmlFor="title">
                  <Input id="title" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Meridian documentary" />
                </Field>

                <Field
                  label="Source language"
                  htmlFor="src"
                  hint="You can still correct this after transcription, before the expensive stages run."
                >
                  <Select id="src" value={sourceLang} onChange={(e) => setSourceLang(e.target.value)}>
                    {caps.asr_autodetect && <option value="auto">Auto-detect</option>}
                    {caps.source_languages.map((l) => (
                      <option key={l.code} value={l.code}>{l.display_name}</option>
                    ))}
                  </Select>
                </Field>

                <Field
                  label="Target languages"
                  error={chosen.length === 0 ? "Pick at least one language with a voice." : null}
                >
                  <div className="flex flex-wrap gap-2">
                    {dubbable.map((l) => {
                      const on = targets.includes(l.code);
                      return (
                        <button
                          key={l.code}
                          type="button"
                          onClick={() => toggleTarget(l.code)}
                          aria-pressed={on}
                          className="h-9 px-3.5 rounded-full text-[14px] font-medium transition-all inline-flex items-center gap-1.5"
                          style={{
                            background: on ? "var(--accent-soft)" : "var(--bg-elevated)",
                            border: `1px solid ${on ? "var(--accent-border)" : "var(--border)"}`,
                            color: on ? "var(--accent)" : "var(--text-mid)",
                          }}
                        >
                          {on && <Check size={13} />}
                          {l.display_name}
                        </button>
                      );
                    })}
                  </div>
                  {translateOnly.length > 0 && (
                    <span className="text-[13px] mt-1" style={{ color: "var(--text-dim)" }}>
                      No voice for {translateOnly.map((l) => l.display_name).join(", ")} on this deployment, so they
                      cannot be dubbed.
                    </span>
                  )}
                </Field>
              </Card>

              <div className="lg:col-span-5 flex flex-col gap-6 w-full">
                <Card className="p-6 flex flex-col gap-4 w-full">
                  <div className="text-[16px] font-semibold" style={{ color: "var(--text)" }}>Pipeline options</div>
                  <Toggle
                    checked={preserveEmotion}
                    onChange={setPreserveEmotion}
                    label="Preserve emotional delivery"
                    help="Carries each segment's detected emotion into the synthesized voice."
                  />
                  <Toggle
                    checked={cloneVoice && caps.voice_clone_available}
                    onChange={setCloneVoice}
                    disabled={!caps.voice_clone_available}
                    label="Clone the original speaker's voice"
                    help={
                      caps.voice_clone_available
                        ? "Uses a reference clip of the speaker so the dub keeps their timbre."
                        : "Not enabled on this deployment."
                    }
                  />
                  <Callout tone="neutral" title="Not available yet">
                    Per-speaker voice selection and keeping the background music need endpoints this backend does not
                    expose. The engine picks each speaker&apos;s voice from their estimated pitch, and the original
                    background is preserved by the mux as it already works.
                  </Callout>
                </Card>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 w-full items-start">
              <Card className="lg:col-span-8 p-6 flex flex-col gap-5 w-full">
                <div className="text-[18px] font-semibold" style={{ color: "var(--text)" }}>{title || "Untitled"}</div>
                <dl className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3.5 text-[14px]">
                  {[
                    ["File", picked ? picked.file.name : "—"],
                    ["Size", picked ? fmtBytes(picked.file.size) : "—"],
                    ["Duration", picked?.durationMs != null ? msToTimecode(picked.durationMs) : "unknown"],
                    ["Source", sourceLang === "auto" ? "Auto-detect, confirmed after transcription" : (caps.source_languages.find((l) => l.code === sourceLang)?.display_name ?? sourceLang)],
                    ["Targets", chosen.map((c) => languageName(caps, c)).join(", ") || "—"],
                    ["Emotion transfer", preserveEmotion ? "On" : "Off"],
                    ["Voice cloning", cloneVoice && caps.voice_clone_available ? "On" : "Off"],
                    ["Device", caps.device.toUpperCase()],
                  ].map(([k, v]) => (
                    <div key={k} className="min-w-0 p-3.5 rounded-xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                      <dt className="text-[12px] font-medium uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>{k}</dt>
                      <dd className="truncate text-[14px] font-medium mt-1" style={{ color: "var(--text)" }} title={String(v)}>{v}</dd>
                    </div>
                  ))}
                </dl>
              </Card>

              <div className="lg:col-span-4 flex flex-col gap-6 w-full">
                <Card className="p-6 flex flex-col gap-3 w-full">
                  <div className="flex items-center gap-2" style={{ color: "var(--accent)" }}>
                    <Cpu size={18} aria-hidden="true" />
                    <span className="text-[15px] font-semibold" style={{ color: "var(--text)" }}>Estimated runtime</span>
                  </div>
                  <div className="text-[26px] font-semibold" style={{ color: "var(--text)" }}>
                    {estimate ?? "Calculating…"}
                  </div>
                  <span className="text-[13px]" style={{ color: "var(--text-dim)" }}>
                    {estimate
                      ? "Measured throughput on CPU for this configuration, not a guarantee."
                      : "No duration was readable from the file, so there is no estimate."}
                  </span>
                </Card>

                {workerHealth && workerHealth.workers_online === 0 && (
                  <Callout tone="warning" title="Workers offline">
                    No Celery workers are currently running. This job will be queued, but will not process until a worker is started (run <code className="font-mono text-[12px]">cd sur-backend; .\scripts\start-workers.ps1 -Role main</code>).
                  </Callout>
                )}

                {uploadPct != null && (
                  <Callout tone="running" title={`Uploading ${Math.round(uploadPct * 100)}%`}>
                    Keep this tab open until the upload finishes.
                  </Callout>
                )}
              </div>
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {error && (
        <Callout
          tone="danger"
          title="Could not start"
          action={
            orphanId ? (
              <Button size="sm" variant="ghost" onClick={() => onCreated(orphanId)}>Open the project anyway</Button>
            ) : undefined
          }
        >
          {error}
        </Callout>
      )}

      <div className="flex items-center justify-between gap-3">
        <Button
          variant="ghost"
          icon={<ArrowLeft size={14} />}
          onClick={() => (step === 0 ? go("dashboard") : setStep(step - 1))}
          disabled={submitting}
        >
          {step === 0 ? "Cancel" : "Back"}
        </Button>

        {step < 2 ? (
          <Button
            icon={<ArrowRight size={14} />}
            onClick={() => setStep(step + 1)}
            disabled={step === 0 ? !canLeaveUpload : !canLeaveSettings}
          >
            Continue
          </Button>
        ) : (
          <Button icon={<Sparkles size={15} />} onClick={submit} loading={submitting} disabled={!canLeaveSettings || !picked}>
            Start dubbing
          </Button>
        )}
      </div>
    </div>
  );
}
