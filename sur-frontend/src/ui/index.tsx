/**
 * VoiceCast UI primitives.
 *
 * Every colour here comes from a CSS variable in index.css, so the light/dark
 * toggle is one attribute on <html> and no component needs to know which
 * theme is active. Nothing in this file fetches, so each primitive stays
 * usable in any state the screens have to render: loading, empty, error, done.
 */
import { AlertTriangle, CheckCircle2, Info, Loader2, X, XCircle } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

// ── Button ───────────────────────────────────────────────────────────────
type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const SIZES: Record<Size, string> = {
  sm: "h-9 px-3.5 text-[13px] gap-1.5",
  md: "h-11 px-5 text-[14px] gap-2",
  lg: "h-13 px-7 text-[15px] gap-2.5",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  icon?: ReactNode;
}

export function Button({
  variant = "primary",
  size = "md",
  loading = false,
  icon,
  children,
  className = "",
  disabled,
  ...rest
}: ButtonProps) {
  const styles: Record<Variant, React.CSSProperties> = {
    primary: { background: "var(--accent-strong)", color: "var(--accent-contrast)", border: "1px solid transparent" },
    secondary: { background: "var(--surface)", color: "var(--text)", border: "1px solid var(--border-strong)" },
    ghost: { background: "transparent", color: "var(--text-mid)", border: "1px solid var(--border)" },
    danger: { background: "var(--danger-soft)", color: "var(--danger)", border: "1px solid var(--danger-border)" },
  };
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={`inline-flex items-center justify-center rounded-lg font-medium transition-all duration-150
        hover:brightness-110 active:scale-[0.98] disabled:opacity-45 disabled:cursor-not-allowed
        disabled:hover:brightness-100 disabled:active:scale-100 whitespace-nowrap ${SIZES[size]} ${className}`}
      style={{ ...styles[variant], ...rest.style }}
    >
      {loading ? <Loader2 size={size === "sm" ? 13 : 15} className="animate-spin" /> : icon}
      {children}
    </button>
  );
}

// ── Card ─────────────────────────────────────────────────────────────────
export function Card({
  children,
  className = "",
  interactive = false,
  as: Tag = "div",
  ...rest
}: { children: ReactNode; className?: string; interactive?: boolean; as?: "div" | "section" | "li" } & React.HTMLAttributes<HTMLElement>) {
  return (
    <Tag
      {...rest}
      className={`rounded-xl transition-all duration-200 ${interactive ? "hover:-translate-y-0.5" : ""} ${className}`}
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        boxShadow: "var(--shadow-sm)",
        ...rest.style,
      }}
      onMouseEnter={interactive ? (e) => { e.currentTarget.style.boxShadow = "var(--shadow-lg)"; e.currentTarget.style.borderColor = "var(--border-strong)"; } : rest.onMouseEnter}
      onMouseLeave={interactive ? (e) => { e.currentTarget.style.boxShadow = "var(--shadow-sm)"; e.currentTarget.style.borderColor = "var(--border)"; } : rest.onMouseLeave}
    >
      {children}
    </Tag>
  );
}

export function CardHeader({ title, action, sub }: { title: ReactNode; action?: ReactNode; sub?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 px-6 py-5" style={{ borderBottom: "1px solid var(--border)" }}>
      <div className="min-w-0">
        <div className="text-[16px] font-semibold truncate" style={{ color: "var(--text)" }}>{title}</div>
        {sub && <div className="text-[13px] mt-1" style={{ color: "var(--text-muted)" }}>{sub}</div>}
      </div>
      {action}
    </div>
  );
}

// ── StatusBadge ──────────────────────────────────────────────────────────
/** One tone per meaning, matching the status colours in index.css. */
export type Tone = "success" | "running" | "warning" | "danger" | "neutral";

const TONE_VARS: Record<Tone, { fg: string; bg: string; bd: string }> = {
  success: { fg: "var(--success)", bg: "var(--success-soft)", bd: "var(--success-border)" },
  running: { fg: "var(--running)", bg: "var(--running-soft)", bd: "var(--running-border)" },
  warning: { fg: "var(--warning)", bg: "var(--warning-soft)", bd: "var(--warning-border)" },
  danger: { fg: "var(--danger)", bg: "var(--danger-soft)", bd: "var(--danger-border)" },
  neutral: { fg: "var(--text-muted)", bg: "var(--surface-hover)", bd: "var(--border)" },
};

export function StatusBadge({ tone, children, pulse = false }: { tone: Tone; children: ReactNode; pulse?: boolean }) {
  const v = TONE_VARS[tone];
  return (
    <span
      className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[12px] font-medium whitespace-nowrap"
      style={{ color: v.fg, background: v.bg, border: `1px solid ${v.bd}` }}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${pulse ? "animate-pulse" : ""}`}
        style={{ background: v.fg }}
        aria-hidden="true"
      />
      {children}
    </span>
  );
}

// ── Callout ──────────────────────────────────────────────────────────────
const CALLOUT_ICON: Record<Tone, typeof Info> = {
  success: CheckCircle2,
  running: Info,
  warning: AlertTriangle,
  danger: XCircle,
  neutral: Info,
};

export function Callout({ tone, title, children, action }: { tone: Tone; title?: ReactNode; children?: ReactNode; action?: ReactNode }) {
  const v = TONE_VARS[tone];
  const Icon = CALLOUT_ICON[tone];
  return (
    <div
      role={tone === "danger" ? "alert" : "status"}
      className="flex gap-3 rounded-xl px-4.5 py-3.5"
      style={{ background: v.bg, border: `1px solid ${v.bd}` }}
    >
      <Icon size={16} style={{ color: v.fg, flexShrink: 0, marginTop: 1 }} aria-hidden="true" />
      <div className="flex flex-col gap-1 min-w-0 flex-1 text-[14px]">
        {title && <span className="font-semibold" style={{ color: v.fg }}>{title}</span>}
        {children && <span style={{ color: "var(--text-mid)" }}>{children}</span>}
        {action}
      </div>
    </div>
  );
}

// ── ProgressBar ──────────────────────────────────────────────────────────
export function ProgressBar({ value, tone = "running", height = 6, label }: { value: number; tone?: Tone; height?: number; label?: string }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label ?? "Progress"}
      className="w-full rounded-full overflow-hidden"
      style={{ height, background: "var(--surface-hover)" }}
    >
      <motion.div
        className="h-full rounded-full"
        style={{ background: TONE_VARS[tone].fg }}
        initial={false}
        animate={{ width: `${pct}%` }}
        transition={{ type: "spring", stiffness: 120, damping: 20 }}
      />
    </div>
  );
}

// ── Skeleton ─────────────────────────────────────────────────────────────
/** A shape where content will be, never a blank screen. */
export function Skeleton({ className = "", rounded = "rounded-lg" }: { className?: string; rounded?: string }) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse ${rounded} ${className}`}
      style={{ background: "var(--surface-hover)" }}
    />
  );
}

export function SkeletonText({ lines = 3, className = "" }: { lines?: number; className?: string }) {
  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className="h-3" rounded="rounded" />
      ))}
    </div>
  );
}

// ── EmptyState ───────────────────────────────────────────────────────────
export function EmptyState({ icon, title, body, action }: { icon?: ReactNode; title: string; body?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center text-center gap-3 py-16 px-6">
      {icon && (
        <div
          className="w-12 h-12 rounded-xl flex items-center justify-center"
          style={{ background: "var(--accent-soft)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
        >
          {icon}
        </div>
      )}
      <h2 className="text-xl font-semibold" style={{ color: "var(--text)", fontFamily: "Inter, sans-serif", letterSpacing: 0 }}>{title}</h2>
      {body && <p className="text-[14px] max-w-md" style={{ color: "var(--text-muted)" }}>{body}</p>}
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
}

// ── Inputs ───────────────────────────────────────────────────────────────
const FIELD = `w-full rounded-lg px-4 text-[14px] outline-none transition-colors`;
const fieldStyle: React.CSSProperties = {
  background: "var(--bg-elevated)",
  border: "1px solid var(--border)",
  color: "var(--text)",
};

export function Field({ label, hint, error, children, htmlFor }: { label: string; hint?: ReactNode; error?: string | null; children: ReactNode; htmlFor?: string }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={htmlFor} className="text-[13px] font-medium" style={{ color: "var(--text-mid)" }}>{label}</label>
      {children}
      {error ? (
        <span className="text-[13px]" style={{ color: "var(--danger)" }}>{error}</span>
      ) : hint ? (
        <span className="text-[13px]" style={{ color: "var(--text-dim)" }}>{hint}</span>
      ) : null}
    </div>
  );
}

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={`${FIELD} h-11 ${props.className ?? ""}`} style={{ ...fieldStyle, ...props.style }} />;
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={`${FIELD} h-11 ${props.className ?? ""}`} style={{ ...fieldStyle, ...props.style }} />;
}

export function Textarea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={`${FIELD} py-2.5 resize-y leading-relaxed ${props.className ?? ""}`} style={{ ...fieldStyle, ...props.style }} />;
}

export function Toggle({ checked, onChange, label, help, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode; help?: ReactNode; disabled?: boolean }) {
  const id = useId();
  return (
    <div className="flex items-start gap-3">
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className="relative w-10 h-6 rounded-full flex-shrink-0 transition-colors disabled:opacity-45 disabled:cursor-not-allowed mt-0.5"
        style={{
          background: checked ? "var(--accent)" : "var(--surface-hover)",
          border: `1px solid ${checked ? "var(--accent)" : "var(--border-strong)"}`,
        }}
      >
        <motion.span
          className="absolute top-0.5 w-4 h-4 rounded-full"
          style={{ background: "#fff" }}
          animate={{ left: checked ? 20 : 3 }}
          transition={{ type: "spring", stiffness: 500, damping: 32 }}
        />
      </button>
      <label htmlFor={id} className="flex flex-col gap-0.5 cursor-pointer select-none">
        <span className="text-[14px] font-medium" style={{ color: "var(--text)" }}>{label}</span>
        {help && <span className="text-[13px]" style={{ color: "var(--text-muted)" }}>{help}</span>}
      </label>
    </div>
  );
}

// ── Toasts ───────────────────────────────────────────────────────────────
interface Toast {
  id: number;
  tone: Tone;
  message: string;
}

const ToastCtx = createContext<(tone: Tone, message: string) => void>(() => {});

export function useToast() {
  return useContext(ToastCtx);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const next = useRef(1);

  const push = useCallback((tone: Tone, message: string) => {
    const id = next.current++;
    setToasts((t) => [...t, { id, tone, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
  }, []);

  return (
    <ToastCtx.Provider value={push}>
      {children}
      {/* aria-live so a screen reader hears the result of an action it cannot see. */}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none" aria-live="polite" aria-atomic="false">
        <AnimatePresence initial={false}>
          {toasts.map((t) => {
            const v = TONE_VARS[t.tone];
            const Icon = CALLOUT_ICON[t.tone];
            return (
              <motion.div
                key={t.id}
                layout
                initial={{ opacity: 0, y: 12, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 8, scale: 0.96 }}
                transition={{ duration: 0.18 }}
                className="pointer-events-auto flex items-start gap-2.5 rounded-xl px-4 py-3 max-w-sm"
                style={{ background: "var(--bg-elevated)", border: `1px solid ${v.bd}`, boxShadow: "var(--shadow-lg)" }}
              >
                <Icon size={15} style={{ color: v.fg, flexShrink: 0, marginTop: 1 }} aria-hidden="true" />
                <span className="text-[13px] flex-1" style={{ color: "var(--text)" }}>{t.message}</span>
                <button
                  onClick={() => setToasts((x) => x.filter((y) => y.id !== t.id))}
                  aria-label="Dismiss notification"
                  style={{ color: "var(--text-dim)" }}
                >
                  <X size={13} />
                </button>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </ToastCtx.Provider>
  );
}

// ── ConfirmDialog ────────────────────────────────────────────────────────
/** Destructive actions ask first. Escape and the backdrop both cancel;
 *  focus moves to the dialog so the keyboard is not left behind it. */
export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = "Confirm",
  tone = "danger",
  busy = false,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  body?: ReactNode;
  confirmLabel?: string;
  tone?: Tone;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    panel.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.14 }}
          style={{ background: "rgba(0,0,0,0.6)" }}
          onClick={onCancel}
        >
          <motion.div
            ref={panel}
            tabIndex={-1}
            role="dialog"
            aria-modal="true"
            aria-label={title}
            initial={{ scale: 0.95, y: 8 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.97, y: 4 }}
            transition={{ duration: 0.16 }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-md rounded-xl p-5 flex flex-col gap-4 outline-none"
            style={{ background: "var(--bg-elevated)", border: "1px solid var(--border-strong)", boxShadow: "var(--shadow-lg)" }}
          >
            <h2 className="text-base font-semibold" style={{ color: "var(--text)", fontFamily: "Inter, sans-serif", letterSpacing: 0 }}>{title}</h2>
            {body && <div className="text-[13px]" style={{ color: "var(--text-mid)" }}>{body}</div>}
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={onCancel} disabled={busy}>Cancel</Button>
              <Button variant={tone === "danger" ? "danger" : "primary"} onClick={onConfirm} loading={busy}>{confirmLabel}</Button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ── Theme ────────────────────────────────────────────────────────────────
export type ThemeMode = "dark" | "light";
const THEME_KEY = "voicecast.theme";

export function useThemeMode(): [ThemeMode, (m: ThemeMode) => void] {
  const [mode, setMode] = useState<ThemeMode>(() => {
    try {
      return localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark";
    } catch {
      return "dark";
    }
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", mode);
    try {
      localStorage.setItem(THEME_KEY, mode);
    } catch {
      /* private mode: the toggle still works for this session */
    }
  }, [mode]);

  return [mode, setMode];
}

// ── small helpers used by several screens ────────────────────────────────
export function msToTimecode(ms: number): string {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = m.toString().padStart(2, "0");
  const ss = sec.toString().padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function fmtDuration(ms: number | null | undefined): string {
  if (!ms || ms <= 0) return "—";
  const s = Math.round(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m ${s % 60}s`;
}

export function fmtWhen(iso: string): string {
  const d = new Date(iso);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
        " " +
        d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function fmtBytes(bytes: number | null | undefined): string {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const mb = bytes / 1024 / 1024;
  return mb < 1 ? `${(bytes / 1024).toFixed(0)} KB` : mb < 1024 ? `${mb.toFixed(1)} MB` : `${(mb / 1024).toFixed(2)} GB`;
}

export { TONE_VARS };

/** Status → tone, in one place so every screen agrees on what green means. */
export function statusTone(status: string, stalled = false): { tone: Tone; label: string; pulse: boolean } {
  if (stalled) return { tone: "warning", label: "Stalled", pulse: false };
  switch (status) {
    case "ready":
      return { tone: "success", label: "Ready", pulse: false };
    case "failed":
      return { tone: "danger", label: "Failed", pulse: false };
    case "processing":
      return { tone: "running", label: "Processing", pulse: true };
    case "queued":
      return { tone: "running", label: "Queued", pulse: true };
    case "awaiting_language_confirmation":
      return { tone: "warning", label: "Needs review", pulse: false };
    case "uploading":
      return { tone: "warning", label: "Uploading", pulse: true };
    case "running":
      return { tone: "running", label: "Running", pulse: true };
    case "pending":
      return { tone: "neutral", label: "Pending", pulse: false };
    default:
      return { tone: "neutral", label: status.replace(/_/g, " "), pulse: false };
  }
}

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    try {
      return window.matchMedia(query).matches;
    } catch {
      return false;
    }
  });
  useEffect(() => {
    // Guarded for the same reason the read above is: an environment without
    // matchMedia (jsdom, some embedded webviews) must get the desktop layout,
    // not a crash that takes the whole screen down.
    if (typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia(query);
    const on = () => setMatches(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [query]);
  return matches;
}

export const useStableMemo = useMemo;
