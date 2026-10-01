/**
 * Voices and Settings.
 *
 * Voices is in the navigation because the brief asks for it, and it shows
 * exactly what /api/capabilities publishes about the engine: which languages
 * have a voice, the engine name, the licences and any release warnings. It
 * does NOT list individual voices or offer previews, because there is no
 * endpoint for either -- no voice catalogue, no sample audio. A grid of
 * invented voice names with dead play buttons would be worse than saying so.
 *
 * Settings is read-only for the same reason: the API publishes capabilities
 * but accepts no settings. The one real preference here (the theme) is a
 * client concern and does belong to the browser.
 */
import { Check, Cpu, Info, Mic2, Moon, ShieldCheck, Sun, X } from "lucide-react";

import { useReadyCapabilities } from "@/lib/capabilities";
import { Callout, Card, CardHeader, StatusBadge, Toggle, type ThemeMode } from "@/ui";

export function Voices() {
  const caps = useReadyCapabilities();
  const withVoice = caps.languages.filter((l) => l.tts_available);
  const without = caps.languages.filter((l) => !l.tts_available);

  return (
    <div className="p-5 md:p-7 flex flex-col gap-5 max-w-[1100px] mx-auto w-full">
      <div>
        <h1 className="text-[26px] font-semibold" style={{ color: "var(--text)" }}>Voices</h1>
        <p className="text-[13px] mt-1" style={{ color: "var(--text-muted)" }}>
          What this deployment can speak, as the engine reports it
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card className="p-4 flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: "var(--accent-soft)", color: "var(--accent)" }} aria-hidden="true">
            <Mic2 size={17} />
          </div>
          <div>
            <div className="text-[12px]" style={{ color: "var(--text-muted)" }}>Engine</div>
            <div className="text-[15px] font-semibold" style={{ color: "var(--text)" }}>{caps.tts_engine}</div>
          </div>
        </Card>
        <Card className="p-4 flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: "var(--surface-hover)", color: "var(--text-muted)" }} aria-hidden="true">
            <Cpu size={17} />
          </div>
          <div>
            <div className="text-[12px]" style={{ color: "var(--text-muted)" }}>Runs on</div>
            <div className="text-[15px] font-semibold" style={{ color: "var(--text)" }}>{caps.device.toUpperCase()}</div>
          </div>
        </Card>
        <Card className="p-4 flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center"
            style={{
              background: caps.tts_commercial_use ? "var(--success-soft)" : "var(--warning-soft)",
              color: caps.tts_commercial_use ? "var(--success)" : "var(--warning)",
            }}
            aria-hidden="true"
          >
            <ShieldCheck size={17} />
          </div>
          <div className="min-w-0">
            <div className="text-[12px]" style={{ color: "var(--text-muted)" }}>Licence</div>
            <div className="text-[15px] font-semibold truncate" style={{ color: "var(--text)" }} title={caps.tts_licenses.join(", ")}>
              {caps.tts_commercial_use ? "Commercial OK" : "Non-commercial"}
            </div>
          </div>
        </Card>
      </div>

      <Card>
        <CardHeader title="Languages" sub={`${withVoice.length} with a voice, ${without.length} without`} />
        <ul className="divide-y" style={{ borderColor: "var(--border)" }}>
          {caps.languages.map((l) => (
            <li key={l.code} className="flex items-center justify-between gap-3 px-5 py-3" style={{ borderTop: "1px solid var(--border)" }}>
              <div className="min-w-0">
                <div className="text-[13px] font-medium" style={{ color: "var(--text)" }}>{l.display_name}</div>
                <div className="text-[11px]" style={{ color: "var(--text-dim)" }}>{l.code} · {l.flores_code}</div>
              </div>
              {l.tts_available ? (
                <StatusBadge tone="success">voice available</StatusBadge>
              ) : (
                <StatusBadge tone="neutral">translate only</StatusBadge>
              )}
            </li>
          ))}
        </ul>
      </Card>

      {caps.tts_voice_warnings.length > 0 && (
        <Callout tone="warning" title="Voice release warnings">
          <ul className="list-disc pl-4 flex flex-col gap-0.5">
            {caps.tts_voice_warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </Callout>
      )}

      <Callout tone="neutral" title="No per-voice catalogue or preview">
        The API publishes which languages have a voice, not the voices themselves, and serves no sample audio. A
        speaker&apos;s voice is chosen by the engine from their estimated pitch. Previewing and picking a voice need
        new endpoints.
      </Callout>
    </div>
  );
}

export function Settings({ theme, setTheme }: { theme: ThemeMode; setTheme: (m: ThemeMode) => void }) {
  const caps = useReadyCapabilities();

  const rows: [string, string | boolean][] = [
    ["TTS engine", caps.tts_engine],
    ["Device", caps.device.toUpperCase()],
    ["Commercial use permitted", caps.tts_commercial_use],
    ["Voice cloning available", caps.voice_clone_available],
    ["ASR auto-detect", caps.asr_autodetect],
    ["Emotion confidence floor", String(caps.emotion_confidence_floor)],
    ["Stalled after", `${caps.stall_after_seconds}s without a heartbeat`],
    ["Max upload", `${caps.max_upload_mb} MB`],
    ["Accepted formats", caps.accepted_formats.join(", ") || "—"],
    ["Emotion labels", caps.emotions.map((e) => e.label).join(", ") || "—"],
    ["Licences", caps.tts_licenses.join(", ") || "—"],
  ];

  return (
    <div className="p-5 md:p-7 flex flex-col gap-5 max-w-[900px] mx-auto w-full">
      <div>
        <h1 className="text-[26px] font-semibold" style={{ color: "var(--text)" }}>Settings</h1>
        <p className="text-[13px] mt-1" style={{ color: "var(--text-muted)" }}>Appearance, and what this deployment reports</p>
      </div>

      <Card>
        <CardHeader title="Appearance" />
        <div className="p-5">
          <Toggle
            checked={theme === "light"}
            onChange={(v) => setTheme(v ? "light" : "dark")}
            label={
              <span className="inline-flex items-center gap-2">
                {theme === "light" ? <Sun size={14} /> : <Moon size={14} />}
                {theme === "light" ? "Light theme" : "Dark theme"}
              </span>
            }
            help="Dark is the default. Stored in this browser only."
          />
        </div>
      </Card>

      <Card>
        <CardHeader
          title="Deployment"
          sub="Read from /api/capabilities"
          action={<Info size={15} style={{ color: "var(--text-dim)" }} aria-hidden="true" />}
        />
        <dl>
          {rows.map(([k, v]) => (
            <div key={k} className="flex items-center justify-between gap-4 px-5 py-2.5" style={{ borderTop: "1px solid var(--border)" }}>
              <dt className="text-[13px]" style={{ color: "var(--text-muted)" }}>{k}</dt>
              <dd className="text-[13px] text-right max-w-[55%] truncate" style={{ color: "var(--text)" }} title={String(v)}>
                {typeof v === "boolean" ? (
                  v ? <Check size={15} style={{ color: "var(--success)" }} aria-label="yes" /> : <X size={15} style={{ color: "var(--text-dim)" }} aria-label="no" />
                ) : (
                  v
                )}
              </dd>
            </div>
          ))}
        </dl>
      </Card>

      <Callout tone="neutral" title="Read-only">
        These come from the backend and cannot be changed from here: the API publishes capabilities but accepts no
        settings. Changing them means changing the deployment&apos;s environment.
      </Callout>
    </div>
  );
}
