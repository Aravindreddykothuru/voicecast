/**
 * The studio chrome: a collapsible left sidebar on desktop, a bottom nav on
 * mobile, and a top bar carrying search, the job-queue indicator, the theme
 * toggle and the user menu.
 *
 * The queue count is real: it comes from the projects the caller already
 * fetched for the dashboard, so the indicator cannot disagree with the list
 * underneath it. The API dot is the same /healthz poll the old header used --
 * it replaced a hardcoded "Core: Online" that stayed green with the backend
 * down, and that lesson is kept here.
 */
import {
  AlertTriangle,
  FolderOpen,
  LayoutDashboard,
  LogOut,
  Menu,
  Mic2,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Search,
  Settings as SettingsIcon,
  Sun,
  X,
} from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { StatusBadge, useMediaQuery, type ThemeMode } from "@/ui";
import type { StoredAuth } from "@/lib/auth";
import type { WorkerHealth } from "@/lib/types";

export type Screen =
  | "dashboard"
  | "new-project"
  | "projects"
  | "voices"
  | "settings"
  | "processing"
  | "editor"
  | "preview"
  | "export";

export const NAV: { key: Screen; label: string; icon: typeof LayoutDashboard }[] = [
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "new-project", label: "New Dub", icon: Plus },
  { key: "projects", label: "Projects", icon: FolderOpen },
  { key: "voices", label: "Voices", icon: Mic2 },
  { key: "settings", label: "Settings", icon: SettingsIcon },
];

const SIDEBAR_KEY = "voicecast.sidebar";

function Logo({ compact }: { compact: boolean }) {
  return (
    <div className="flex items-center gap-2.5 min-w-0">
      <div
        className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
        style={{ background: "var(--accent)", color: "#fff" }}
        aria-hidden="true"
      >
        <Mic2 size={18} />
      </div>
      {!compact && (
        <span
          className="truncate"
          style={{ fontFamily: "Sora, sans-serif", fontWeight: 600, fontSize: 16, letterSpacing: "0.06em", color: "var(--text)" }}
        >
          VOICECAST
        </span>
      )}
    </div>
  );
}

function NavButton({
  item,
  active,
  compact,
  onClick,
}: {
  item: (typeof NAV)[number];
  active: boolean;
  compact: boolean;
  onClick: () => void;
}) {
  const Icon = item.icon;
  return (
    <button
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      title={compact ? item.label : undefined}
      className={`relative flex items-center gap-3 h-11 rounded-lg transition-colors w-full ${compact ? "justify-center px-0" : "px-3.5"}`}
      style={{
        color: active ? "var(--text)" : "var(--text-muted)",
        background: active ? "var(--accent-soft)" : "transparent",
      }}
      onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "var(--surface-hover)"; }}
      onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}
    >
      {active && (
        <motion.span
          layoutId="nav-active"
          className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-full"
          style={{ background: "var(--accent)" }}
          aria-hidden="true"
        />
      )}
      <Icon size={19} style={{ flexShrink: 0, color: active ? "var(--accent)" : undefined }} aria-hidden="true" />
      {!compact && <span className="text-[14px] font-medium truncate">{item.label}</span>}
    </button>
  );
}

export function AppShell({
  screen,
  go,
  auth,
  onLogout,
  apiUp,
  queue,
  theme,
  setTheme,
  search,
  setSearch,
  workerHealth,
  showWorkerBanner = false,
  onDismissWorkerBanner,
  children,
}: {
  screen: Screen;
  go: (s: Screen) => void;
  auth: StoredAuth | null;
  onLogout: () => void;
  apiUp: boolean | null;
  queue: { running: number; queued: number } | null;
  theme: ThemeMode;
  setTheme: (m: ThemeMode) => void;
  search: string;
  setSearch: (v: string) => void;
  workerHealth?: WorkerHealth | null;
  showWorkerBanner?: boolean;
  onDismissWorkerBanner?: () => void;
  children: ReactNode;
}) {
  const isMobile = useMediaQuery("(max-width: 767px)");
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(SIDEBAR_KEY) === "1";
    } catch {
      return false;
    }
  });
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try {
      localStorage.setItem(SIDEBAR_KEY, collapsed ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [collapsed]);

  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setMenuOpen(false); };
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("mousedown", onDown); window.removeEventListener("keydown", onKey); };
  }, [menuOpen]);

  const compact = collapsed && !isMobile;
  const initials = (() => {
    const label = auth?.user.name?.trim() || auth?.user.email || "";
    const parts = label.split(/\s+/).filter(Boolean);
    if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
    return label.slice(0, 2).toUpperCase() || "?";
  })();

  const busy = (queue?.running ?? 0) + (queue?.queued ?? 0);

  return (
    <div className="flex h-full w-full" style={{ background: "var(--bg)", color: "var(--text)" }}>
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:top-2 focus:left-2 focus:px-3 focus:py-2 focus:rounded-lg"
        style={{ background: "var(--accent)", color: "#fff" }}
      >
        Skip to content
      </a>

      {/* Sidebar — desktop only; mobile gets the bottom nav instead. */}
      {!isMobile && (
        <aside
          className="flex flex-col flex-shrink-0 transition-all duration-200"
          style={{ width: compact ? 68 : 230, borderRight: "1px solid var(--border)", background: "var(--bg-elevated)" }}
          aria-label="Primary"
        >
          <div className={`h-16 flex items-center ${compact ? "justify-center" : "justify-between px-5"} flex-shrink-0`} style={{ borderBottom: "1px solid var(--border)" }}>
            <Logo compact={compact} />
          </div>

          <nav className="flex-1 flex flex-col gap-1 p-2.5 overflow-y-auto">
            {NAV.map((item) => (
              <NavButton key={item.key} item={item} active={screen === item.key} compact={compact} onClick={() => go(item.key)} />
            ))}
          </nav>

          <div className="p-2.5" style={{ borderTop: "1px solid var(--border)" }}>
            <button
              onClick={() => setCollapsed(!collapsed)}
              aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
              className={`flex items-center gap-3 h-9 rounded-lg w-full transition-colors ${compact ? "justify-center" : "px-3"}`}
              style={{ color: "var(--text-dim)" }}
              onMouseEnter={(e) => { e.currentTarget.style.background = "var(--surface-hover)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
            >
              {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
              {!compact && <span className="text-[13px]">Collapse</span>}
            </button>
          </div>
        </aside>
      )}

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top bar */}
        <header
          className="h-16 flex items-center gap-3 px-5 sm:px-7 flex-shrink-0"
          style={{ borderBottom: "1px solid var(--border)", background: "var(--bg-elevated)" }}
        >
          {isMobile && <Logo compact={false} />}

          <div className="relative flex-1 max-w-xl hidden sm:block">
            <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: "var(--text-dim)" }} aria-hidden="true" />
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search projects…"
              aria-label="Search projects"
              className="w-full h-10 rounded-lg pl-10 pr-3 text-[14px] outline-none transition-colors"
              style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--text)" }}
            />
          </div>

          <div className="flex-1 sm:hidden" />

          {/* Job queue indicator — real counts, or nothing. */}
          {queue && (
            <button
              onClick={() => go("projects")}
              className="hidden sm:inline-flex"
              aria-label={`${queue.running} running, ${queue.queued} queued. Open projects.`}
            >
              {busy > 0 ? (
                <StatusBadge tone="running" pulse>
                  {queue.running} running · {queue.queued} queued
                </StatusBadge>
              ) : (
                <StatusBadge tone="neutral">Queue idle</StatusBadge>
              )}
            </button>
          )}

          <span
            className="hidden md:inline-flex items-center gap-1.5 text-[12px]"
            style={{ color: apiUp === false ? "var(--danger)" : "var(--text-muted)" }}
            title="Backend reachability, re-checked every 30s"
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: apiUp === false ? "var(--danger)" : apiUp ? "var(--success)" : "var(--text-dim)" }}
              aria-hidden="true"
            />
            API {apiUp === null ? "checking" : apiUp ? "online" : "offline"}
          </span>

          <span
            className="hidden md:inline-flex items-center gap-1.5 text-[12px]"
            style={{
              color:
                workerHealth === null
                  ? "var(--text-dim)"
                  : workerHealth.workers_online > 0
                  ? "var(--text-muted)"
                  : "var(--danger)",
            }}
            title={
              workerHealth === null
                ? "Worker reachability, checking…"
                : `${workerHealth.workers_online} worker(s) online (polled every 12s)`
            }
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background:
                  workerHealth === null
                    ? "var(--text-dim)"
                    : workerHealth.workers_online > 0
                    ? "var(--success)"
                    : "var(--danger)",
              }}
              aria-hidden="true"
            />
            Workers {workerHealth === null ? "checking" : workerHealth.workers_online > 0 ? "online" : "offline"}
          </span>

          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            className="w-10 h-10 rounded-lg flex items-center justify-center transition-colors flex-shrink-0"
            style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}
            onMouseEnter={(e) => { e.currentTarget.style.background = "var(--surface-hover)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
          >
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
          </button>

          {/* User menu */}
          <div className="relative flex-shrink-0" ref={menuRef}>
            <button
              onClick={() => setMenuOpen((o) => !o)}
              aria-haspopup="menu"
              aria-expanded={menuOpen}
              // WCAG 2.5.3: the accessible name has to CONTAIN the visible
              // label. The visible label is the initials, so "User menu"
              // alone failed label-content-name-mismatch -- and hiding the
              // initials from the tree did not help, because the check reads
              // what is on screen.
              aria-label={`${initials}, user menu`}
              className="w-10 h-10 rounded-full flex items-center justify-center text-[14px] font-semibold"
              style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-border)", color: "var(--accent)" }}
            >
              {initials}
            </button>
            <AnimatePresence>
              {menuOpen && (
                <motion.div
                  role="menu"
                  initial={{ opacity: 0, y: -6, scale: 0.97 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -4, scale: 0.98 }}
                  transition={{ duration: 0.14 }}
                  className="absolute right-0 top-11 w-56 rounded-xl overflow-hidden z-40"
                  style={{ background: "var(--bg-elevated)", border: "1px solid var(--border-strong)", boxShadow: "var(--shadow-lg)" }}
                >
                  <div className="px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                    <div className="text-[14px] font-medium truncate" style={{ color: "var(--text)" }}>{auth?.user.name || "Signed in"}</div>
                    <div className="text-[13px] truncate" style={{ color: "var(--text-muted)" }}>{auth?.user.email}</div>
                  </div>
                  <button
                    role="menuitem"
                    onClick={() => { setMenuOpen(false); go("settings"); }}
                    className="w-full flex items-center gap-2.5 px-4 py-2.5 text-[14px] text-left transition-colors"
                    style={{ color: "var(--text-mid)" }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "var(--surface-hover)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                  >
                    <SettingsIcon size={15} /> Settings
                  </button>
                  <button
                    role="menuitem"
                    onClick={() => { setMenuOpen(false); onLogout(); }}
                    className="w-full flex items-center gap-2.5 px-4 py-2.5 text-[14px] text-left transition-colors"
                    style={{ color: "var(--danger)" }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "var(--danger-soft)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                  >
                    <LogOut size={15} /> Log out
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </header>

        {showWorkerBanner && (
          <div
            role="alert"
            className="flex items-center justify-between px-6 py-2.5 text-[13px] border-b flex-shrink-0"
            style={{
              background: "rgba(239, 68, 68, 0.12)",
              borderColor: "rgba(239, 68, 68, 0.25)",
              color: "var(--danger)",
            }}
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <AlertTriangle size={15} className="flex-shrink-0" />
              <span className="truncate">
                No workers are running. Jobs will not process until a worker is started. Start command: <code className="px-1.5 py-0.5 rounded font-mono text-[12px]" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>cd sur-backend; .\scripts\start-workers.ps1 -Role main</code>
              </span>
            </div>
            <button
              onClick={onDismissWorkerBanner}
              aria-label="Dismiss worker warning banner"
              className="p-1 rounded hover:opacity-80 transition-opacity ml-3 flex-shrink-0"
            >
              <X size={15} />
            </button>
          </div>
        )}

        <main id="main" className="flex-1 overflow-y-auto w-full min-w-0" style={{ paddingBottom: isMobile ? 76 : 0 }} aria-label="Main content">
          {children}
        </main>

        {/* Mobile bottom nav */}
        {isMobile && (
          <nav
            className="fixed bottom-0 left-0 right-0 h-[68px] flex items-stretch z-30"
            style={{ background: "var(--bg-elevated)", borderTop: "1px solid var(--border)" }}
            aria-label="Primary"
          >
            {NAV.map((item) => {
              const Icon = item.icon;
              const active = screen === item.key;
              return (
                <button
                  key={item.key}
                  onClick={() => go(item.key)}
                  aria-current={active ? "page" : undefined}
                  className="flex-1 flex flex-col items-center justify-center gap-1"
                  style={{ color: active ? "var(--accent)" : "var(--text-dim)" }}
                >
                  <Icon size={21} aria-hidden="true" />
                  <span className="text-[11px] font-medium">{item.label}</span>
                </button>
              );
            })}
          </nav>
        )}
      </div>
    </div>
  );
}

export { Menu, X };
