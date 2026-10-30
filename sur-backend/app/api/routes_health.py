from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
import redis
from sqlalchemy import select

from app.celery_app import celery_app
from app.config import get_settings
from app.db import session_scope
from app.models.project import Project, ProjectStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

CACHE_TTL_SECONDS = 5.0
_cache: dict[str, Any] = {"expires_at": 0.0, "data": None}


class WorkerHealthResponse(BaseModel):
    workers_online: int
    queues: dict[str, int]
    oldest_queued_age_seconds: int | None


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/api/health/workers", response_model=WorkerHealthResponse)
@router.get("/health/workers", response_model=WorkerHealthResponse)
def get_worker_health() -> WorkerHealthResponse:
    """Worker health check (PRD / Reliability requirements BUG 2).
    Returns number of workers online, queue depths, and age of oldest queued project.
    Cached for a few seconds so polling doesn't hammer the broker.
    """
    global _cache
    now_ts = time.time()
    if _cache["data"] is not None and now_ts < _cache["expires_at"]:
        return _cache["data"]

    # 1. Ping Celery workers with 1.5s timeout
    workers_online = 0
    try:
        inspect = celery_app.control.inspect(timeout=1.5)
        ping_res = inspect.ping()
        if ping_res:
            workers_online = len(ping_res)
    except Exception as e:
        logger.warning("Worker health inspect.ping() failed: %s", e)

    # 2. Query Redis queue depths directly from broker
    settings = get_settings()
    queues = {
        "q.extract_audio": 0,
        "q.chunk_and_diarize": 0,
        "q.transcribe": 0,
        "q.detect_emotion": 0,
        "q.translate": 0,
        "q.synthesize": 0,
        "q.mux_export": 0,
    }
    try:
        r = redis.Redis.from_url(settings.celery_broker_url)
        for q in queues.keys():
            try:
                queues[q] = r.llen(q)
            except Exception:
                pass
    except Exception as e:
        logger.warning("Could not query Redis queue lengths: %s", e)

    # 3. Oldest queued project age + heartbeat fallback for busy solo workers
    oldest_age: int | None = None
    recent_heartbeat_found = False
    try:
        with session_scope() as db:
            oldest_p = db.execute(
                select(Project)
                .where(Project.status == ProjectStatus.queued)
                .order_by(Project.updated_at.asc())
            ).scalars().first()
            if oldest_p and oldest_p.updated_at:
                now_dt = datetime.now(timezone.utc)
                age = (now_dt - oldest_p.updated_at).total_seconds()
                oldest_age = max(0, int(age))

            if workers_online == 0:
                now_dt = datetime.now(timezone.utc)
                active_projects = db.execute(
                    select(Project).where(
                        Project.status.in_([ProjectStatus.processing, ProjectStatus.queued]),
                        Project.heartbeat_at.isnot(None),
                    )
                ).scalars().all()
                for p in active_projects:
                    if p.heartbeat_at and (now_dt - p.heartbeat_at).total_seconds() < 35:
                        recent_heartbeat_found = True
                        break
    except Exception as e:
        logger.warning("Could not query oldest queued age or heartbeats: %s", e)

    if workers_online == 0 and recent_heartbeat_found:
        workers_online = 1

    result = WorkerHealthResponse(
        workers_online=workers_online,
        queues=queues,
        oldest_queued_age_seconds=oldest_age,
    )
    _cache = {"expires_at": now_ts + CACHE_TTL_SECONDS, "data": result}
    return result
