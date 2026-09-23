"""Run a job to completion, surviving whatever happens to it.

  * Restart recovery: lines left RENDERING/CHECKING go back to PENDING;
    attempts left running are marked interrupted; every DONE/FLAGGED output
    is re-hashed and re-rendered if missing or changed; stray temp files swept.
  * Each line: failover.render_line, then the chosen audio converted to the
    unified format and written atomically; the line's final state, model,
    key and output hash committed in one transaction.
  * SIGINT/SIGTERM (and SIGBREAK on Windows): the current line finishes, state
    is persisted, the job is marked INTERRUPTED, exit is clean.
    `tts resume <job>` continues from there.
  * Disk: checked before every line; below the threshold the job pauses with
    an alert and resumes when space returns. ENOSPC mid-write deletes the
    temp file, puts the line back to PENDING and pauses -- nothing corrupted.
  * Heartbeat file every 2 s and an optional HTTP health endpoint.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path

from app.tts_runtime import consistency, failover
from app.tts_runtime.adapters import adapter_class
from app.tts_runtime.audio import to_unified, write_wav_atomic
from app.tts_runtime.breaker import CircuitBreaker
from app.tts_runtime.checks import CALIBRATION_PATH, Calibration, CheckResult
from app.tts_runtime.config import RuntimeConfig
from app.tts_runtime.fsutil import (
    DiskFull,
    atomic_write_bytes,
    free_mb,
    sha256_file,
    sweep_temp_files,
)
from app.tts_runtime.knownbad import DEFAULT_DIR as KNOWN_BAD_DIR
from app.tts_runtime.knownbad import KnownBad
from app.tts_runtime.netmon import ConnectivityMonitor
from app.tts_runtime.netqueue import TaskQueue
from app.tts_runtime.pool import ModelPool
from app.tts_runtime.store import Store

logger = logging.getLogger(__name__)


def job_id_for(spec: dict) -> str:
    return spec.get("job_id") or hashlib.sha256(
        json.dumps(spec.get("lines", []), ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


class Runner:
    def __init__(self, cfg: RuntimeConfig, home: str | Path, *, calibration: Calibration | None = None,
                 options: dict | None = None, env_extra: dict | None = None, netmon: ConnectivityMonitor | None = None,
                 known_bad_dir: str | Path = KNOWN_BAD_DIR, alert=None, disk_poll_s: float = 5.0,
                 free_mb_fn=free_mb, task_handlers: dict | None = None, health_port: int | None = None,
                 heartbeat_s: float = 2.0):
        self.cfg = cfg
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.home / "runtime.sqlite3")
        if calibration is None:
            cal_path = Path(os.environ.get("TTS_RUNTIME_CALIBRATION") or CALIBRATION_PATH)
            if cal_path.exists():
                calibration = Calibration.load(cal_path)
            else:
                logger.warning("no calibration file at %s: duration/silence triggers are off for this run", cal_path)
        self.calibration = calibration
        self._options = options or {}
        self.env_extra = dict(env_extra or {})
        if cfg.offline_mode != "off":
            self.env_extra.setdefault("HF_HUB_OFFLINE", "1")
            self.env_extra.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.alerts: list[tuple[str, str]] = []
        self._alert_hook = alert
        self.netmon = netmon or ConnectivityMonitor(cfg.offline_mode)
        self.breaker = CircuitBreaker(self.store, cfg.breaker, self.alert)
        self.knownbad = KnownBad(self.store, cfg.known_bad.hits, cfg.known_bad.of_last, known_bad_dir)
        self.pool = ModelPool(cfg, self.home, self.options_for, self.store, env_extra=self.env_extra)
        from app.tts_runtime.provision import download_handler

        self.tasks = TaskQueue(self.store, self.netmon, cfg.network_backoff_s,
                               {"download": download_handler(self.home), **(task_handlers or {})})
        self._refetching: dict[str, set[str]] = {}
        self.disk_poll_s = disk_poll_s
        self.free_mb_fn = free_mb_fn
        self.health_port = health_port
        self.heartbeat_s = heartbeat_s
        self._stop = threading.Event()
        self.current: dict = {}
        self.last_error: str | None = None
        self._health = None

    # --- plumbing -------------------------------------------------------------
    def options_for(self, model_id: str) -> dict:
        base = {"carrier": self.cfg.uses_carrier(model_id), "home": str(self.home)}
        base.update(self._options.get(model_id, {}))
        return base

    def alert(self, kind: str, msg: str) -> None:
        self.alerts.append((kind, msg))
        logger.warning("ALERT %s: %s", kind, msg)
        if self._alert_hook:
            try:
                self._alert_hook(kind, msg)
            except Exception:  # noqa: BLE001
                logger.exception("alert hook failed")

    def request_stop(self, *_):
        if not self._stop.is_set():
            logger.warning("stop requested: finishing the current line, then persisting state")
            self.store.event("stop_requested", self.current.get("job"), detail="signal")
        self._stop.set()

    def _install_signals(self) -> dict:
        old = {}
        if threading.current_thread() is not threading.main_thread():
            return old
        sigs = [signal.SIGINT, signal.SIGTERM] + ([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else [])
        for s in sigs:
            try:
                old[s] = signal.signal(s, self.request_stop)
            except (ValueError, OSError):
                pass
        return old

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            self.write_heartbeat()

    def write_heartbeat(self) -> None:
        hb = {"pid": os.getpid(), "at": time.time(), **self.current,
              "network": self.netmon.state, "last_error": self.last_error}
        try:
            atomic_write_bytes(self.home / "heartbeat.json", json.dumps(hb, ensure_ascii=False).encode())
        except OSError:
            pass

    # --- jobs -------------------------------------------------------------------
    def submit(self, spec: dict) -> str:
        from app.tts_runtime.adapters import adapter_class

        lines = spec.get("lines") or []
        for i, ln in enumerate(lines):
            if not isinstance(ln.get("text"), str) or not ln.get("lang"):
                raise ValueError(f"line {i}: needs text and lang")
            # A line no model in its chain could ever speak must not enter the
            # queue: it would be flagged on every run, forever. This is the
            # static claim (`supports`), not availability -- a model that is
            # merely unprovisioned is a runtime skip, logged and recoverable,
            # whereas Urdu (dropped 2026-09-23) has no model that speaks it at
            # all. `app.capabilities` is deliberately not consulted here: the
            # runtime also renders languages the dub product does not target,
            # such as English for the parler_tiny test model.
            lang = ln["lang"]
            chain = self.cfg.chain_for(lang)
            if not any(adapter_class(m).supports(lang) for m in chain):
                raise ValueError(
                    f"line {i}: no model in the {lang!r} chain speaks {lang!r} "
                    f"(chain: {', '.join(chain) or 'empty'}). Nothing would ever render it.")
        jid = job_id_for(spec)
        out_dir = str(Path(spec.get("out_dir") or self.home / "jobs" / jid))
        if self.store.create_job(jid, spec, out_dir, lines):
            self.store.event("job_created", jid, detail={"lines": len(lines)})
        return jid

    def recover(self, job_id: str) -> dict:
        job = self.store.job(job_id)
        if job is None:
            raise KeyError(f"no job {job_id}")
        out = {"requeued": [], "invalid_outputs": [], "swept": 0}
        out["swept"] = sweep_temp_files(job["out_dir"]) + sweep_temp_files(self.home / "attempts")
        with self.store.tx() as db:
            db.execute("UPDATE attempts SET outcome='interrupted', finished_at=? WHERE job_id=? AND outcome='running'",
                       (time.time(), job_id))
        for ln in self.store.lines(job_id, ("RENDERING", "CHECKING")):
            self.store.set_line(job_id, ln["idx"], "PENDING")
            out["requeued"].append(ln["idx"])
        for ln in self.store.lines(job_id, ("DONE", "FLAGGED")):
            p = ln["output_path"]
            if p is None and ln["state"] == "FLAGGED":
                continue     # flagged with no audio at all: nothing to verify
            if p is None or not Path(p).exists() or sha256_file(p) != ln["output_sha256"]:
                self.store.set_line(job_id, ln["idx"], "PENDING", output_path=None, output_sha256=None)
                out["invalid_outputs"].append(ln["idx"])
        if out["requeued"] or out["invalid_outputs"]:
            self.store.event("restart_recovered", job_id, detail=out)
            logger.warning("job %s recovered after interruption: %s", job_id, out)
        return out

    def _disk_guard(self, job_id: str) -> bool:
        """True to continue; waits (paused) while disk is low. False if stopped."""
        paused = False
        while not self._stop.is_set():
            free = self.free_mb_fn(self.home)
            if free >= self.cfg.min_free_disk_mb:
                if paused:
                    self.store.set_job_status(job_id, "RUNNING")
                    self.store.event("resume", job_id, detail={"reason": "disk space available", "free_mb": round(free)})
                return True
            if not paused:
                paused = True
                msg = f"free disk {free:.0f} MB < {self.cfg.min_free_disk_mb} MB; batch paused, no file touched"
                self.store.set_job_status(job_id, "PAUSED", "disk_low")
                self.store.event("pause", job_id, detail={"reason": "disk_low", "free_mb": round(free)})
                self.alert("disk_low", msg)
            self._stop.wait(self.disk_poll_s)
        return False

    def _write_output(self, job, idx: int, res: failover.LineResult) -> tuple[str | None, str | None]:
        if res.chosen is None or not res.chosen.has_audio:
            return None, None
        x, sr = failover.audio_of(res.chosen)
        path = Path(job["out_dir"]) / f"{idx:05d}.wav"
        sha = write_wav_atomic(path, to_unified(x, sr))
        return str(path), sha

    def run(self, job_id: str) -> dict:
        job = self.store.job(job_id)
        if job is None:
            raise KeyError(f"no job {job_id}")
        old_signals = self._install_signals()
        self._stop.clear()
        rec = self.recover(job_id)
        self.store.set_job_status(job_id, "RUNNING")
        self.store.event("job_started", job_id, detail={"recovered": rec,
                                                         "test_models": self.cfg.allow_test_models})
        self.current = {"job": job_id}
        self.netmon.start()
        self.tasks.start()
        hb = threading.Thread(target=self._heartbeat, name="heartbeat", daemon=True)
        hb.start()
        if self.health_port is not None:
            from app.tts_runtime.health import serve

            self._health = serve(self, self.health_port)
        ctx = failover.Context(self.cfg, self.store, self.pool, self.breaker, self.knownbad, self.calibration,
                               self.home, self.options_for, job_id)
        ctx.on_corrupt = self.on_corrupt
        try:
            while not self._stop.is_set():
                pending = self.store.lines(job_id, ("PENDING",))
                if not pending:
                    break
                ln = pending[0]
                if not self._disk_guard(job_id):
                    break
                self._restore_refetched(ctx)
                line = {k: ln[k] for k in ("idx", "text", "lang", "speaker", "emotion", "scene")}
                self.current.update(idx=ln["idx"], state="RENDERING")
                self.write_heartbeat()
                res = failover.render_line(ctx, line)
                try:
                    path, sha = self._write_output(job, ln["idx"], res)
                except DiskFull as e:
                    self.last_error = str(e)
                    self.store.set_line(job_id, ln["idx"], "PENDING")
                    self.store.event("disk_full", job_id, idx=ln["idx"], detail=str(e))
                    self.alert("disk_full", f"{e}; line {ln['idx']} back to PENDING")
                    continue
                c = res.chosen
                self.store.set_line(job_id, ln["idx"], res.state, model=c.model if c else None,
                                    model_version=c.version if c else None, key=c.key if c else None,
                                    output_path=path, output_sha256=sha, flag_reason=res.reason or None)
                if res.state == "FLAGGED":
                    self.store.event("flagged", job_id, lang=line["lang"], idx=ln["idx"],
                                     detail={"text": line["text"], "reason": res.reason,
                                             "kept": c.model if c else None})
                    logger.warning("line %s FLAGGED for review: %s", ln["idx"], res.reason)
                if res.unverified:
                    self.store.event("unverified", job_id, idx=ln["idx"], detail="reader unavailable")
            stopped = self._stop.is_set() and bool(self.store.lines(job_id, ("PENDING",)))
            if not stopped and self.cfg.scene_consistency:
                self._unify_scenes(ctx, job)
            status = "INTERRUPTED" if stopped else "DONE"
            self.store.set_job_status(job_id, status)
            self.store.event("job_" + status.lower(), job_id, detail=self.store.counts(job_id))
            from app.tts_runtime.report import write_report

            write_report(self.store, job_id, Path(job["out_dir"]), self.pool.status())
            return {"job": job_id, "status": status, "counts": self.store.counts(job_id), "recovered": rec}
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self.store.set_job_status(job_id, "INTERRUPTED", self.last_error[:200])
            self.store.event("runner_error", job_id, detail=self.last_error)
            raise
        finally:
            self.current.update(state="stopped")
            self.write_heartbeat()
            self._stop.set()
            self.tasks.stop()
            self.netmon.stop()
            self.pool.close_all()
            if self._health:
                self._health.shutdown()
            for s, h in old_signals.items():
                try:
                    signal.signal(s, h)
                except (ValueError, OSError):
                    pass

    def _unify_scenes(self, ctx, job) -> None:
        rows = {r["idx"]: r for r in self.store.lines(job["id"]) if r["model"]}
        results = {}
        for idx, r in rows.items():
            line = {k: r[k] for k in ("idx", "text", "lang", "speaker", "emotion", "scene")}
            t = failover.Try(r["model"], r["model_version"] or "", r["key"] or "", 0, "ok", CheckResult(True))
            results[idx] = (line, failover.LineResult(r["state"], t))
        before = {i: res.chosen.model for i, (_l, res) in results.items()}
        consistency.unify(ctx, results)
        for idx, (_line, res) in results.items():
            if res.chosen and res.chosen.model != before.get(idx) and res.chosen.has_audio:
                path, sha = self._write_output(job, idx, res)
                c = res.chosen
                self.store.set_line(job["id"], idx, "DONE", model=c.model, model_version=c.version, key=c.key,
                                    output_path=path, output_sha256=sha, flag_reason=None)

    def on_corrupt(self, model: str, path: str) -> None:
        """A model file failed its hash at load: delete it and queue a verified re-fetch."""
        try:
            os.remove(path)
        except OSError:
            pass
        from app.tts_runtime.provision import enqueue_refetch

        before = {t["id"] for t in self.store.tasks(("QUEUED",))}
        enqueue_refetch(self.store, model, path, self.options_for(model))
        queued = {t["id"] for t in self.store.tasks(("QUEUED",))} - before
        self._refetching.setdefault(model, set()).update(queued or {t["id"] for t in self.store.tasks(("QUEUED",))})
        self.alert("model_corrupt", f"{model}: {path} failed its hash; deleted, re-fetch queued, never loaded")

    def _restore_refetched(self, ctx) -> None:
        """A model whose corrupt file has been re-fetched and verified comes back."""
        for model, ids in list(self._refetching.items()):
            states = {r["state"] for r in self.store.tasks() if r["id"] in ids}
            if states and states <= {"DONE"}:
                ctx.reset_availability(model)
                self.store.event("model_restored", self.current.get("job"), model=model,
                                 detail="corrupt file re-fetched and verified")
                logger.warning("model %s restored: re-fetched file verified", model)
                del self._refetching[model]

    def model_status(self) -> dict:
        out = {}
        for model in self.cfg.models():
            ok, why = adapter_class(model).availability(self.options_for(model))
            br = {r["lang"]: r["state"] for r in self.store.q("SELECT lang, state FROM breakers WHERE model=?", (model,))}
            out[model] = {"available": ok, "why": why, "breakers": br}
        return out
