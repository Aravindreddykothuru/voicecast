"""Is a run still making progress, or is it stuck?

The failure this exists for: a project in "queued"/"processing" whose work
will never happen -- no worker consuming the queue, a worker killed mid-stage,
a chain that was never fully enqueued. Nothing marks such a run failed, so
the UI showed a spinner forever.

Liveness is a heartbeat, not "time since the row last changed": CPU stages
legitimately run for many minutes without touching the project row
(diarization is one blocking call), so inactivity alone can't tell slow from
dead. Every pipeline task runs inside `heartbeat(project_id)`, a daemon thread
that stamps projects.heartbeat_at every HEARTBEAT_INTERVAL_SECONDS. Model
code releases the GIL (torch, ctranslate2, onnxruntime), so the thread keeps
beating through long native calls and stops when the process dies.
"""
from __future__ import annotations

import contextlib
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select, text

from app.config import get_settings

logger = logging.getLogger(__name__)

# Statuses in which some worker is supposed to be doing something.
ACTIVE_STATUSES = frozenset({"queued", "processing"})


def touch(project_id: str) -> None:
    """Stamp the heartbeat with a raw UPDATE: going through the ORM would also
    bump updated_at, which means "the project changed", not "still alive"."""
    from app.db import engine

    with engine.begin() as conn:
        conn.execute(text("UPDATE projects SET heartbeat_at = now() WHERE id = :id"), {"id": project_id})


@contextlib.contextmanager
def heartbeat(project_id: str):
    interval = get_settings().heartbeat_interval_seconds
    stop = threading.Event()

    def beat() -> None:
        while True:
            try:
                touch(project_id)
            except Exception as exc:  # noqa: BLE001 -- a missed beat must never fail the task
                logger.warning("heartbeat for %s failed: %s", project_id, exc)
            if stop.wait(interval):
                return

    thread = threading.Thread(target=beat, name=f"heartbeat-{project_id[:8]}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval + 5)


# Every stage task takes project_id as its first argument. Heartbeats are
# started and stopped by Celery's own task signals rather than by each task
# body, so a new stage cannot forget to report that it is alive.
PIPELINE_TASKS = frozenset({
    "app.pipeline.tasks.extract_audio",
    "app.pipeline.tasks.chunk_and_diarize",
    "app.pipeline.tasks.transcribe",
    "app.pipeline.tasks.detect_emotion",
    "app.pipeline.tasks.translate",
    "app.pipeline.tasks.synthesize",
    "app.pipeline.tasks.mux_export",
})
_running: dict[str, contextlib.ExitStack] = {}


def start_task_heartbeat(task_id: str, task_name: str, args) -> None:
    from app.models.base import is_uuid

    if task_name not in PIPELINE_TASKS or not args or not is_uuid(args[0]):
        return
    stack = contextlib.ExitStack()
    stack.enter_context(heartbeat(args[0]))
    _running[task_id] = stack


def stop_task_heartbeat(task_id: str) -> None:
    stack = _running.pop(task_id, None)
    if stack is not None:
        stack.close()


def connect_signals() -> None:
    from celery.signals import task_postrun, task_prerun

    @task_prerun.connect(weak=False)
    def _prerun(task_id=None, task=None, args=None, **_):
        start_task_heartbeat(task_id, task.name, args)

    @task_postrun.connect(weak=False)
    def _postrun(task_id=None, **_):
        stop_task_heartbeat(task_id)


@dataclass(frozen=True)
class RunLiveness:
    last_activity_at: datetime | None
    stalled: bool
    # Why the UI should say it looks stuck, in words an operator can act on.
    stalled_reason: str | None


def last_activity_at(db, project) -> datetime | None:
    from app.models.export_job import ExportJob
    from app.models.segment import Segment

    latest_segment = db.execute(
        select(func.max(Segment.updated_at)).where(Segment.project_id == project.id)
    ).scalar_one_or_none()
    latest_export = db.execute(
        select(func.max(ExportJob.updated_at)).where(ExportJob.project_id == project.id)
    ).scalar_one_or_none()
    stamps = [t for t in (project.heartbeat_at, project.updated_at, latest_segment, latest_export) if t is not None]
    return max(stamps) if stamps else None


def run_liveness(db, project, *, now: datetime | None = None) -> RunLiveness:
    last = last_activity_at(db, project)
    status = project.status.value if hasattr(project.status, "value") else str(project.status)
    if status not in ACTIVE_STATUSES or last is None:
        return RunLiveness(last, False, None)

    now = now or datetime.now(timezone.utc)
    quiet = (now - last).total_seconds()
    limit = get_settings().stall_after_seconds
    if quiet <= limit:
        return RunLiveness(last, False, None)

    minutes = int(quiet // 60)
    if status == "queued" or project.heartbeat_at is None or project.heartbeat_at <= project.updated_at:
        reason = (
            f"No worker has picked this run up in {minutes} min. Check that the "
            "Celery workers are running and consuming the pipeline queues."
        )
    else:
        stage = project.current_stage or "the current stage"
        reason = (
            f"The worker running {stage} stopped reporting {minutes} min ago -- it "
            "was probably stopped or crashed. Restarting resumes from the last finished segment."
        )
    return RunLiveness(last, True, reason)
