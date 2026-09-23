"""Parent side of the worker processes: start, talk, time out, recover.

At most `max_loaded` TTS models are resident; loading another unloads the
least recently used. A request that outlives its timeout kills the worker
(a hung model cannot be interrupted any other way); a worker that dies is a
crash with its exit code and the tail of its stderr log. Out-of-memory
follows the ladder in the brief: free the cache (the worker does), unload the
least recently used other model, retry once, then restart on CPU if the model
supports it -- otherwise the caller moves to the next model.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

from app.tts_runtime.config import BACKEND_DIR, RuntimeConfig

logger = logging.getLogger(__name__)


class WorkerError(RuntimeError):
    kind = "exception"

    def __init__(self, model_id: str, detail: str):
        super().__init__(f"{model_id}: {detail}")
        self.model_id = model_id
        self.detail = detail


class WorkerTimeout(WorkerError):
    kind = "timeout"


class WorkerCrashed(WorkerError):
    kind = "crash"


class WorkerOOM(WorkerError):
    kind = "oom"


class WorkerCorrupt(WorkerError):
    kind = "corrupt"

    def __init__(self, model_id, detail, path):
        super().__init__(model_id, detail)
        self.path = path


class WorkerUnavailable(WorkerError):
    kind = "unavailable"


def _tail(path: Path, n: int = 1500) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


class WorkerHandle:
    def __init__(self, model_id: str, options: dict, interpreter: str | None, log_dir: Path,
                 env_extra: dict | None = None, allow_test_models: bool = False):
        self.model_id = model_id
        self.options = options
        self.interpreter = interpreter or sys.executable
        self.log_path = log_dir / f"worker-{model_id.replace(':', '_')}-{os.getpid()}-{int(time.time() * 1000)}.log"
        self.env_extra = env_extra or {}
        self.allow_test_models = allow_test_models
        self.proc: subprocess.Popen | None = None
        self.device: str | None = None
        self.load_info: dict = {}
        self._q: queue.Queue = queue.Queue()

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update({
            "TTS_WORKER_SPEC": json.dumps({"model_id": self.model_id, "options": self.options,
                                           "allow_test_models": self.allow_test_models}),
            "PYTHONPATH": str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", ""),
            "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
        })
        env.update(self.env_extra)
        self._log = open(self.log_path, "ab")
        # Own process group / session: a Ctrl+C aimed at the runner must not
        # kill the worker mid-line -- the runner finishes the line, then stops.
        kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
              else {"start_new_session": True})
        self.proc = subprocess.Popen([self.interpreter, "-u", "-m", "app.tts_runtime.worker"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                     cwd=str(BACKEND_DIR), env=env, **kw)
        q, out = self._q, self.proc.stdout

        def pump():
            for line in iter(out.readline, b""):
                q.put(line)
            q.put(None)   # EOF: the worker is gone

        threading.Thread(target=pump, name=f"pump-{self.model_id}", daemon=True).start()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def request(self, msg: dict, timeout: float) -> dict:
        if not self.alive():
            raise WorkerCrashed(self.model_id, f"worker not running (exit {self.proc.poll() if self.proc else None})")
        try:
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            self.proc.stdin.flush()
        except OSError as e:
            raise WorkerCrashed(self.model_id, f"worker pipe closed: {e}") from None
        try:
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            self.kill()
            raise WorkerTimeout(self.model_id, f"{msg.get('op')} exceeded {timeout:.0f}s; worker killed") from None
        if line is None:
            code = self.proc.wait(timeout=10)
            raise WorkerCrashed(self.model_id, f"worker exited with code {code} during {msg.get('op')}; "
                                               f"stderr tail: {_tail(self.log_path)[-600:]}")
        return json.loads(line)

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def close(self, timeout: float = 10.0) -> None:
        if self.alive():
            try:
                self.request({"op": "shutdown"}, timeout)
            except WorkerError:
                pass
        self.kill()
        try:
            self._log.close()
        except Exception:  # noqa: BLE001
            pass


class ModelPool:
    def __init__(self, cfg: RuntimeConfig, home: Path, options_for, store=None, device: str = "auto",
                 env_extra: dict | None = None, load_timeout_s: float = 900.0):
        self.cfg = cfg
        self.home = home
        self.options_for = options_for
        self.store = store
        self.device = device
        self.env_extra = env_extra or {}
        self.load_timeout_s = load_timeout_s
        self._workers: OrderedDict[str, WorkerHandle] = OrderedDict()
        self._readers: dict[str, WorkerHandle] = {}
        self.cpu_forced: set[str] = set()
        self.stats: dict[str, dict] = {}

    def _event(self, kind, model=None, detail=""):
        if self.store is not None:
            self.store.event(kind, model=model, detail=detail)

    def _spawn(self, model_id: str, device: str) -> WorkerHandle:
        h = WorkerHandle(model_id, self.options_for(model_id), self.cfg.interpreter_for(model_id),
                         self.home / "logs", self.env_extra, self.cfg.allow_test_models)
        if h.interpreter != sys.executable and not Path(h.interpreter).exists():
            raise WorkerUnavailable(model_id, f"interpreter {h.interpreter} does not exist (run make setup-tts)")
        t0 = time.time()
        h.start()
        r = h.request({"op": "load", "device": device}, self.load_timeout_s)
        if not r.get("ok"):
            h.kill()
            self._raise(model_id, r)
        h.device = r.get("device")
        h.load_info = r
        st = self.stats.setdefault(model_id, {})
        st.update(load_s=round(time.time() - t0, 2), rss_mb=r.get("rss_mb"), vram_mb=r.get("vram_mb"),
                  device=h.device)
        logger.info("loaded %s on %s in %.1fs", model_id, h.device, time.time() - t0)
        return h

    def _raise(self, model_id: str, r: dict):
        err = r.get("error")
        detail = r.get("detail", "")
        if err == "oom":
            raise WorkerOOM(model_id, detail)
        if err == "corrupt":
            raise WorkerCorrupt(model_id, detail, r.get("path"))
        if err in ("unavailable", "license"):
            raise WorkerUnavailable(model_id, detail)
        raise WorkerError(model_id, f"{r.get('type', 'error')}: {detail}")

    def get(self, model_id: str) -> WorkerHandle:
        h = self._workers.get(model_id)
        if h is not None and h.alive():
            self._workers.move_to_end(model_id)
            return h
        if h is not None:
            self._workers.pop(model_id).close()
        while len(self._workers) >= max(1, self.cfg.max_models_loaded):
            self.evict_lru(reason="max_models_on_gpu")
        device = "cpu" if model_id in self.cpu_forced else self.options_for(model_id).get("device", self.device)
        h = self._spawn(model_id, device)
        self._workers[model_id] = h
        return h

    def evict_lru(self, except_model: str | None = None, reason: str = "") -> str | None:
        for mid in list(self._workers):
            if mid != except_model:
                self._workers.pop(mid).close()
                self._event("unload", mid, reason)
                logger.info("unloaded %s (%s)", mid, reason)
                return mid
        return None

    def drop(self, model_id: str) -> None:
        h = self._workers.pop(model_id, None)
        if h:
            h.kill()

    def synth(self, model_id: str, text: str, lang: str, speaker, emotion, draw: int, out: str) -> dict:
        """One synthesis with the OOM ladder. Raises a WorkerError subclass on failure."""
        req = {"op": "synth", "text": text, "lang": lang, "speaker": speaker, "emotion": emotion,
               "draw": draw, "out": out}
        h = self._get_with_oom_ladder(model_id)
        r = self._req(h, model_id, req)
        if r.get("error") == "oom":
            self._event("oom", model_id, "freeing cache, unloading least recently used model, retrying once")
            logger.warning("OOM in %s: unloading LRU model and retrying once", model_id)
            self.evict_lru(except_model=model_id, reason=f"OOM in {model_id}")
            r = self._req(self.get(model_id), model_id, req)
            if r.get("error") == "oom":
                cls_ok = self._supports_cpu(model_id)
                if cls_ok and model_id not in self.cpu_forced:
                    self._event("oom_cpu_fallback", model_id, "second OOM; restarting on CPU")
                    logger.warning("second OOM in %s: restarting on CPU", model_id)
                    self.cpu_forced.add(model_id)
                    self.drop(model_id)
                    r = self._req(self.get(model_id), model_id, req)
        if not r.get("ok"):
            self._raise(model_id, r)
        return r

    def _get_with_oom_ladder(self, model_id: str) -> WorkerHandle:
        """Loading can run out of memory too (on a GPU host, that is the common
        case): unload the least recently used model and retry, then CPU."""
        try:
            return self.get(model_id)
        except WorkerOOM:
            self._event("oom", model_id, "out of memory while loading; unloading least recently used model")
            self.evict_lru(except_model=model_id, reason=f"OOM loading {model_id}")
        try:
            return self.get(model_id)
        except WorkerOOM:
            if not self._supports_cpu(model_id) or model_id in self.cpu_forced:
                raise
            self._event("oom_cpu_fallback", model_id, "second OOM while loading; loading on CPU")
            self.cpu_forced.add(model_id)
            return self.get(model_id)

    def _supports_cpu(self, model_id: str) -> bool:
        from app.tts_runtime.adapters import adapter_class

        return bool(adapter_class(model_id).supports_cpu_fallback)

    def _req(self, h: WorkerHandle, model_id: str, req: dict) -> dict:
        try:
            return h.request(req, self.cfg.synth_timeout_s)
        except (WorkerTimeout, WorkerCrashed):
            self._workers.pop(model_id, None)
            raise

    def health(self, model_id: str) -> dict:
        try:
            r = self._get_with_oom_ladder(model_id).request({"op": "health"}, self.cfg.synth_timeout_s)
        except WorkerError as e:
            self._workers.pop(model_id, None)
            return {"ok": False, "kind": e.kind, "detail": f"{e.kind}: {e.detail}",
                    "path": getattr(e, "path", None)}
        return r.get("health", {"ok": False, "detail": r.get("detail", "no reply")})

    # --- readers (separate from the TTS LRU) ------------------------------------
    def read(self, reader: str, wav: str, lang: str) -> str:
        h = self._readers.get(reader)
        if h is None or not h.alive():
            h = WorkerHandle(reader, self.options_for(reader), None, self.home / "logs", self.env_extra)
            h.start()
            r = h.request({"op": "load", "device": "cpu"}, self.load_timeout_s)
            if not r.get("ok"):
                h.kill()
                self._raise(reader, r)
            self._readers[reader] = h
        try:
            r = h.request({"op": "read", "wav": wav, "lang": lang}, self.cfg.check_timeout_s)
        except (WorkerTimeout, WorkerCrashed):
            self._readers.pop(reader, None)
            raise
        if not r.get("ok"):
            self._raise(reader, r)
        return r["text"]

    def status(self) -> dict:
        return {"loaded": {m: {"device": h.device, "alive": h.alive(), "pid": h.proc.pid if h.proc else None}
                           for m, h in self._workers.items()},
                "readers": {m: h.alive() for m, h in self._readers.items()},
                "cpu_forced": sorted(self.cpu_forced), "stats": self.stats}

    def close_all(self) -> None:
        for h in list(self._workers.values()) + list(self._readers.values()):
            h.close()
        self._workers.clear()
        self._readers.clear()
