"""The queue of work that needs the network (model downloads).

Offline, running and queued tasks move to PAUSED; rendering is unaffected.
Online again, PAUSED tasks go back to QUEUED and run on the backoff
schedule (2, 4, 8 ... 300 s, capped), resuming partial downloads where they
stopped. Every pause, resume and retry is an event in the store and counts
in the run report. Nobody has to do anything.
"""
from __future__ import annotations

import json
import logging
import threading
import time

from app.tts_runtime.downloader import AuthRequired, NetworkDown
from app.tts_runtime.manifest import HashMismatch

logger = logging.getLogger(__name__)
MAX_HASH_FAILURES = 3


class TaskQueue:
    def __init__(self, store, netmon, backoff_s, handlers: dict, clock=time.time):
        self.store = store
        self.netmon = netmon
        self.backoff_s = tuple(backoff_s)
        self.handlers = handlers
        self.clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._recovered = False

    def recover(self) -> list[str]:
        """Tasks left RUNNING by a process that died go back to QUEUED. Without
        this a download killed mid-file stays RUNNING forever and its partial
        file is never resumed (found by killing a real 1.27 GB download)."""
        ids = []
        for t in self.store.tasks(("RUNNING",)):
            self.store.set_task(t["id"], "QUEUED", next_at=0, last_error="recovered after interruption")
            self.store.event("task_recovered", detail={"task": t["id"]})
            logger.warning("network task %s was RUNNING when its process died; re-queued", t["id"])
            ids.append(t["id"])
        self._recovered = True
        return ids

    def delay(self, attempts: int) -> float:
        return self.backoff_s[min(max(attempts - 1, 0), len(self.backoff_s) - 1)]

    def pending(self) -> int:
        return len(self.store.tasks(("QUEUED", "RUNNING", "PAUSED")))

    def run_once(self) -> int:
        """One scheduling pass. Returns the number of tasks that ran."""
        if not self._recovered:
            self.recover()
        if not self.netmon.is_up():
            for t in self.store.tasks(("QUEUED", "RUNNING")):
                self.store.set_task(t["id"], "PAUSED", last_error="network down")
                self.store.event("pause", detail={"task": t["id"], "reason": "network down"})
                logger.warning("PAUSED network task %s (offline)", t["id"])
            return 0
        now = self.clock()
        for t in self.store.tasks(("PAUSED",)):
            wait = self.delay(t["attempts"])
            self.store.set_task(t["id"], "QUEUED", next_at=now + wait)
            self.store.event("resume", detail={"task": t["id"], "after_s": wait})
            logger.warning("RESUMING network task %s in %.0fs (network back)", t["id"], wait)
        ran = 0
        for t in self.store.tasks(("QUEUED",)):
            if t["next_at"] > self.clock():
                continue
            if not self.netmon.is_up():
                break
            ran += 1
            self._run(t)
        return ran

    def _run(self, t) -> None:
        tid, attempts = t["id"], t["attempts"]
        self.store.set_task(tid, "RUNNING")
        try:
            self.handlers[t["kind"]](json.loads(t["payload"]))
        except AuthRequired as e:
            self.store.set_task(tid, "FAILED", last_error=str(e), attempts=attempts + 1)
            self.store.event("task_failed", detail={"task": tid, "error": str(e)})
            logger.error("network task %s FAILED: %s", tid, e)
        except NetworkDown as e:
            attempts += 1
            wait = self.delay(attempts)
            self.store.set_task(tid, "QUEUED", attempts=attempts, next_at=self.clock() + wait, last_error=str(e))
            self.store.event("retry", detail={"task": tid, "attempt": attempts, "after_s": wait, "error": str(e)})
            logger.warning("network task %s interrupted (%s); retry %d in %.0fs", tid, e, attempts, wait)
            self.netmon.check_now()
        except HashMismatch as e:
            attempts += 1
            state = "FAILED" if attempts >= MAX_HASH_FAILURES else "QUEUED"
            self.store.set_task(tid, state, attempts=attempts, next_at=self.clock(), last_error=str(e))
            self.store.event("hash_mismatch", detail={"task": tid, "attempt": attempts, "error": str(e)})
            logger.error("network task %s: %s (deleted; %s)", tid, e,
                         "giving up" if state == "FAILED" else "re-fetching")
        except Exception as e:  # noqa: BLE001 -- recorded, never swallowed
            self.store.set_task(tid, "FAILED", attempts=attempts + 1, last_error=f"{type(e).__name__}: {e}")
            self.store.event("task_failed", detail={"task": tid, "error": f"{type(e).__name__}: {e}"})
            logger.exception("network task %s failed", tid)
        else:
            self.store.set_task(tid, "DONE", attempts=attempts + 1, last_error=None)
            self.store.event("task_done", detail={"task": tid})

    def start(self, poll_s: float = 1.0) -> None:
        if self._thread:
            return

        def loop():
            while not self._stop.is_set():
                try:
                    self.run_once()
                except Exception:  # noqa: BLE001
                    logger.exception("task queue pass failed")
                self._stop.wait(poll_s)

        self._thread = threading.Thread(target=loop, name="netqueue", daemon=True)
        self._thread.start()

    def stop(self, join_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(join_s)
