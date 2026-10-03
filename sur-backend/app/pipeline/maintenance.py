"""Background and startup sweeps for pipeline health and maintenance."""
from __future__ import annotations

import logging
from sqlalchemy import select

from app.db import session_scope
from app.models.project import Project, ProjectStatus
from app.models.segment import Segment
from app.pipeline.tasks import has_speech

logger = logging.getLogger(__name__)


def sweep_stuck_projects() -> int:
    """Finds projects stuck in awaiting_language_confirmation with zero segments
    or no valid speech transcribed, and marks them failed with permanent errors.

    Addresses BUG 1: Prevents projects from being stuck in awaiting_language_confirmation
    when diarization or transcription produced nothing.
    """
    cleaned = 0
    with session_scope() as db:
        projects = db.execute(
            select(Project).where(Project.status == ProjectStatus.awaiting_language_confirmation)
        ).scalars().all()

        for p in projects:
            segs = db.execute(select(Segment).where(Segment.project_id == p.id)).scalars().all()
            if not segs:
                p.status = ProjectStatus.failed
                p.current_stage = "transcribe"
                p.error_message = "transcribe: no segments were produced for this project."
                p.error_is_permanent = True
                cleaned += 1
                logger.warning("Swept stuck project %s: 0 segments, marked as failed", p.id)
            elif not any(has_speech(s) for s in segs):
                p.status = ProjectStatus.failed
                p.current_stage = "transcribe"
                p.error_message = "transcribe: no speech was transcribed in any segment -- source may be silent, music-only, or corrupt."
                p.error_is_permanent = True
                cleaned += 1
                logger.warning("Swept stuck project %s: no speech transcribed, marked as failed", p.id)
    return cleaned
