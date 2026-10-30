"""A watchdog that keeps a job running until it is DONE.

Runs `tts resume <job>` as a child. If the child dies (crash, kill -9, OOM
killer) or its heartbeat goes stale (hung), it is killed if needed and
started again -- `resume` recovers the interrupted line itself. A SIGINT or
SIGTERM to the supervisor is forwarded to the child, which finishes its
current line and exits; the supervisor then stops without restarting.

On Linux the same job can run under systemd instead (docs/tts-runtime.md
has a unit file): Restart=on-failure, ExecStart=... tts resume <job>.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from app.tts_runtime.config import BACKEND_DIR
from app.tts_runtime.store import Store

logger = logging.getLogger(__name__)


def heartbeat_age(home: Path) -> float | None:
    p = home / "heartbeat.json"
    try:
        return time.time() - p.stat().st_mtime
    except OSError:
        return None


def supervise(job_id: str, home: Path, *, config: str | None = None, max_restarts: int = 50,
              stale_s: float = 180.0, backoff_s: tuple[float, ...] = (1, 2, 5, 10, 30), poll_s: float = 1.0,
              extra_env: dict | None = None) -> int:
    store = Store(home / "runtime.sqlite3")
    stopping = {"flag": False}
    child: dict = {"p": None}

    def forward(signum, _frame):
        stopping["flag"] = True
        p = child["p"]
        if p and p.poll() is None:
            logger.warning("supervisor: forwarding stop to the runner; it will finish its line and exit")
            try:
                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    p.send_signal(signal.SIGTERM)
            except OSError:
                pass

    for s in [signal.SIGINT, signal.SIGTERM] + ([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else []):
        try:
            signal.signal(s, forward)
        except (ValueError, OSError):
            pass

    restarts = 0
    cmd = [sys.executable, "-m", "app.tts_runtime", "--home", str(home)]
    if config:
        cmd += ["--config", config]
    cmd += ["resume", job_id]
    env = dict(os.environ, **(extra_env or {}))
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    while True:
        kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {})
        p = subprocess.Popen(cmd, cwd=str(BACKEND_DIR), env=env, **kw)
        child["p"] = p
        started = time.time()
        while p.poll() is None:
            time.sleep(poll_s)
            age = heartbeat_age(home)
            if age is not None and time.time() - started > stale_s and age > stale_s:
                logger.error("supervisor: heartbeat stale for %.0fs; killing hung runner", age)
                store.event("supervisor_kill_hung", job_id, detail={"heartbeat_age_s": round(age)})
                p.kill()
                p.wait()
        job = store.job(job_id)
        status = job["status"] if job else "?"
        if status == "DONE":
            store.event("supervisor_done", job_id, detail={"restarts": restarts})
            return 0
        if stopping["flag"]:
            store.event("supervisor_stopped", job_id, detail={"status": status})
            return 3
        restarts += 1
        if restarts > max_restarts:
            store.event("supervisor_gave_up", job_id, detail={"restarts": restarts - 1})
            logger.error("supervisor: %d restarts, giving up (job %s is %s)", restarts - 1, job_id, status)
            return 1
        wait = backoff_s[min(restarts - 1, len(backoff_s) - 1)]
        store.event("supervisor_restart", job_id, detail={"exit": p.returncode, "status": status, "restart": restarts,
                                                          "after_s": wait})
        logger.warning("supervisor: runner exited %s with job %s; restart %d in %.0fs", p.returncode, status,
                       restarts, wait)
        time.sleep(wait)
