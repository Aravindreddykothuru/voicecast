/**
 * VoiceCast — app root.
 *
 * Structure: CapabilitiesProvider wraps everything so the fetch is shared,
 * but the GATE (which blocks on it) wraps only the studio. The landing page
 * renders with the backend down, because it offers no language, voice or
 * emotion and so has nothing to guess. That split is the fix for
 * "the page won't open": the gate used to wrap the whole app, and a stopped
 * backend hid the product entirely. See src/App.gate.test.tsx.
 *
 * The capabilities fetch and the /healthz poll are both deferred until the
 * studio mounts, so the landing page makes no requests and logs no console
 * errors when there is no API to talk to.
 */
import { useCallback, useEffect, useState, type ReactNode } from "react";

import EchoLanding from "@/EchoLanding";
import { AppShell, type Screen } from "@/components/AppShell";
import { apiReachable, getWorkerHealth, listProjects } from "@/lib/api";
import { getStoredAuth, logout, type StoredAuth } from "@/lib/auth";
import { CapabilitiesProvider, useCapabilities } from "@/lib/capabilities";
import type { ProjectListItem, WorkerHealth } from "@/lib/types";
import { Dashboard } from "@/screens/Dashboard";
import { Editor } from "@/screens/Editor";
import { ExportScreen } from "@/screens/ExportScreen";
import { NewDub } from "@/screens/NewDub";
import { ProgressScreen } from "@/screens/Progress";
import { Projects } from "@/screens/Projects";
import { Settings, Voices } from "@/screens/VoicesSettings";
import { Button, Skeleton, ToastProvider, useThemeMode } from "@/ui";

// ── Capabilities gate ─────────────────────────────────────────────────────
// The HARD RULE, scoped to the screens it is about: nothing that offers a
// language, a voice or an emotion renders until /api/capabilities answers, and
// on failure it blocks with a retry rather than a fallback list
// (sur-backend/CONTRACTS.md #2). It does not wrap the landing page.
function FullScreen({ children }: { children: ReactNode }) {
  return (
    <div
      className="min-h-full w-full flex items-center justify-center p-8"
      style={{ background: "var(--bg)", color: "var(--text)" }}
    >
      <div className="max-w-md w-full flex flex-col items-center gap-4 text-center">{children}</div>
    </div>
  );
}

function CapabilitiesGate({ children, onBack }: { children: ReactNode; onBack?: () => void }) {
  const { caps, loading, error, reload } = useCapabilities();

  if (loading) {
    return (
      <div className="page-container gap-6">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 w-full">
          {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
        </div>
        <Skeleton className="h-6 w-40 mt-2" />
        <div className="grid gap-5 w-full" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))" }}>
          {Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-48" />)}
        </div>
        <span className="sr-only" role="status">Loading capabilities</span>
      </div>
    );
  }

  if (error || !caps) {
    return (
      <FullScreen>
        <div style={{ fontFamily: "Sora, sans-serif", fontWeight: 300, fontSize: 24 }}>Cannot reach the backend</div>
        <div className="text-[13px]" style={{ color: "var(--text-muted)" }}>
          {error ?? "The capabilities endpoint returned nothing."}
        </div>
        <div className="text-[12px]" style={{ color: "var(--text-dim)" }}>
          The studio stays blocked rather than guess which languages and emotions the engine supports. Start the API
          (<code>uvicorn app.main:app</code>) and retry.
        </div>
        <div className="flex items-center gap-3">
          <Button onClick={reload}>Retry</Button>
          {onBack && <Button variant="ghost" onClick={onBack}>Back to home</Button>}
        </div>
      </FullScreen>
    );
  }

  return <>{children}</>;
}

// ── Studio ────────────────────────────────────────────────────────────────
function Studio() {
  const [screen, setScreen] = useState<Screen>("dashboard");
  const [landing, setLanding] = useState(() => !getStoredAuth());
  const [projectId, setProjectId] = useState<string | null>(null);
  const [auth, setAuth] = useState<StoredAuth | null>(() => getStoredAuth());
  const [theme, setTheme] = useThemeMode();
  const [search, setSearch] = useState("");

  const [projects, setProjects] = useState<ProjectListItem[] | null>(null);
  const [projectsError, setProjectsError] = useState<string | null>(null);
  const [apiUp, setApiUp] = useState<boolean | null>(null);
  const [workerHealth, setWorkerHealth] = useState<WorkerHealth | null>(null);
  const [workerBannerDismissed, setWorkerBannerDismissed] = useState(false);

  const go = useCallback((s: Screen) => setScreen(s), []);
  const openProject = useCallback((id: string, to: Screen) => {
    setProjectId(id);
    setScreen(to);
  }, []);

  const handleEnter = () => {
    setAuth(getStoredAuth());
    setLanding(false);
    setScreen("dashboard");
  };
  const handleLogout = () => {
    logout();
    setAuth(null);
    setLanding(true);
  };

  // One projects fetch for the whole shell: the dashboard, the Projects screen
  // and the top bar's queue indicator all read it, so they cannot disagree.
  const reloadProjects = useCallback(() => {
    setProjectsError(null);
    listProjects()
      .then(setProjects)
      .catch((e) => setProjectsError(e instanceof Error ? e.message : "Failed to load projects"));
  }, []);

  // Deliberately after the landing branch: no request is made until the
  // studio is actually on screen.
  useEffect(() => {
    if (landing) return;
    reloadProjects();
    const t = setInterval(reloadProjects, 10000);
    return () => clearInterval(t);
  }, [landing, reloadProjects]);

  useEffect(() => {
    if (landing) return;
    let alive = true;
    const check = () => apiReachable().then((ok) => { if (alive) setApiUp(ok); });
    check();
    const t = setInterval(check, 30000);
    return () => { alive = false; clearInterval(t); };
  }, [landing]);

  // Worker health polling (every 12 seconds per BUG 2 requirement: 10-15s)
  useEffect(() => {
    if (landing) return;
    let alive = true;
    const checkWorkers = () =>
      getWorkerHealth()
        .then((h) => {
          if (!alive) return;
          setWorkerHealth(h);
          if (h.workers_online > 0) {
            setWorkerBannerDismissed(false);
          }
        })
        .catch(() => {
          if (alive) setWorkerHealth({ workers_online: 0, queues: {}, oldest_queued_age_seconds: null });
        });
    checkWorkers();
    const t = setInterval(checkWorkers, 12000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [landing]);

  if (landing) return <EchoLanding enter={handleEnter} />;

  const queue = projects
    ? {
        running: projects.filter((p) => p.status === "processing").length,
        queued: projects.filter((p) => ["queued", "uploading", "awaiting_language_confirmation"].includes(p.status)).length,
      }
    : null;

  const workersOffline = workerHealth !== null && workerHealth.workers_online === 0;
  const hasActiveJobs = (queue?.running ?? 0) > 0 || (queue?.queued ?? 0) > 0;
  const showWorkerBanner = workersOffline && hasActiveJobs && !workerBannerDismissed;

  return (
    // The provider lives here, below the landing branch, so the capabilities
    // fetch happens when the studio mounts and not before. On the landing
    // page nothing is requested, which is why it logs no console errors with
    // the backend down.
    <CapabilitiesProvider>
    <AppShell
      screen={screen}
      go={go}
      auth={auth}
      onLogout={handleLogout}
      apiUp={apiUp}
      queue={queue}
      theme={theme}
      setTheme={setTheme}
      search={search}
      setSearch={setSearch}
      workerHealth={workerHealth}
      showWorkerBanner={showWorkerBanner}
      onDismissWorkerBanner={() => setWorkerBannerDismissed(true)}
    >
      <CapabilitiesGate onBack={() => setLanding(true)}>
        {screen === "dashboard" && (
          <Dashboard go={go} openProject={openProject} projects={projects} error={projectsError} reload={reloadProjects} search={search} />
        )}
        {screen === "new-project" && (
          <NewDub go={go} onCreated={(id) => openProject(id, "processing")} workerHealth={workerHealth} />
        )}
        {screen === "projects" && (
          <Projects projects={projects} error={projectsError} reload={reloadProjects} openProject={openProject} go={go} search={search} />
        )}
        {screen === "voices" && <Voices />}
        {screen === "settings" && <Settings theme={theme} setTheme={setTheme} />}
        {screen === "processing" && <ProgressScreen projectId={projectId} go={go} />}
        {screen === "editor" && <Editor projectId={projectId} go={go} />}
        {screen === "export" && <ExportScreen projectId={projectId} go={go} />}
      </CapabilitiesGate>
    </AppShell>
    </CapabilitiesProvider>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <Studio />
    </ToastProvider>
  );
}
