"""Single-segment regeneration -- POST /api/segments/{id}/regenerate.

Deliberately NOT a Celery chain of the full-project tasks: those operate on
"all segments in status X" and would either skip this one segment (if its
status doesn't match) or require resetting its status first (racy against
whatever else is touching the project). Instead this runs the requested
stages against exactly one segment, reusing the same provider-calling
helpers the full-project tasks use, so behavior never drifts between the two
paths.

Split across the queues whose workers own the models:

  regenerate_segment    q.translate   validate, translate if asked, hand off
  regenerate_synthesize q.synthesize  render, then re-mux the export

It used to be one task on q.synthesize that also translated -- that worker
runs the TTS venv, which has no IndicTrans2 -- and it never re-muxed, so a
regenerated line was audible in the editor but the downloadable video still
said the old one. A failure used to emit an event and leave the project
wherever it was; it now marks the project failed like any stage does.
"""
from __future__ import annotations

import logging

from app.celery_app import QUEUE_TRANSLATE, QUEUE_TTS, celery_app
from app.db import session_scope
from app.logging_conf import bind_context, clear_context
from app.models.base import is_uuid
from app.models.project import Project
from app.models.segment import Segment
from app.pipeline.events import emit_segment_ready, emit_stage_completed, emit_stage_started
from app.pipeline.liveness import heartbeat
from app.pipeline.tasks import _mark_project_failed, has_speech, mux_export, synthesize_segment, translate_segment
from app.storage import get_storage

logger = logging.getLogger(__name__)

VALID_STAGES = {"translate", "synthesize"}
_PERMANENT = (ValueError, LookupError, TypeError)
STAGE = "regenerate"


def _load(db, segment_id: str) -> tuple[Segment, Project]:
    # Check before querying: Postgres's uuid type raises DataError on a
    # malformed id, and DataError isn't permanent, so the task would burn
    # three retries with backoff on an id that can never be valid.
    segment = db.get(Segment, segment_id) if is_uuid(segment_id) else None
    if segment is None:
        raise ValueError(f"segment {segment_id} not found")
    return segment, db.get(Project, segment.project_id)


@celery_app.task(
    bind=True,
    name="app.pipeline.regenerate.regenerate_segment",
    queue=QUEUE_TRANSLATE,
    autoretry_for=(Exception,),
    dont_autoretry_for=_PERMANENT,
    max_retries=3,
    retry_backoff=True,
)
def regenerate_segment(self, segment_id: str, stages: list[str]) -> str:
    requested = [s for s in stages if s in VALID_STAGES]
    if "synthesize" not in requested:
        raise ValueError(f"regenerate needs 'synthesize' in its stages, got {stages!r}")

    with session_scope() as db:
        _, project = _load(db, segment_id)
        project_id = project.id

    bind_context(project_id=project_id, segment_id=segment_id)
    emit_stage_started(project_id, STAGE)
    try:
        with heartbeat(project_id):
            if "translate" in requested:
                with session_scope() as db:
                    segment, project = _load(db, segment_id)
                    if not project.target_languages:
                        raise ValueError(f"project {project.id} has no target_languages configured")
                    # Same rule as the full-project translate stage: never
                    # guess a source language. Confirmed project.source_language
                    # wins; this segment's own ASR detection is the fallback.
                    source_lang = project.source_language or segment.detected_language
                    if not source_lang and has_speech(segment):
                        raise ValueError(
                            f"segment {segment.id} has no confirmed or detected source "
                            "language -- cannot translate without guessing."
                        )
                    translate_segment(segment, project.target_languages[0], source_lang or "")
                emit_segment_ready(project_id, segment_id)
        # Its own queue, its own worker; see the module docstring.
        regenerate_synthesize.delay(segment_id)
        return segment_id
    except Exception as exc:  # noqa: BLE001
        _mark_project_failed(project_id, STAGE, exc)
        raise
    finally:
        clear_context()


@celery_app.task(
    bind=True,
    name="app.pipeline.regenerate.regenerate_synthesize",
    queue=QUEUE_TTS,
    autoretry_for=(Exception,),
    dont_autoretry_for=_PERMANENT,
    max_retries=3,
    retry_backoff=True,
)
def regenerate_synthesize(self, segment_id: str) -> str:
    with session_scope() as db:
        _, project = _load(db, segment_id)
        project_id = project.id

    bind_context(project_id=project_id, segment_id=segment_id)
    try:
        with heartbeat(project_id):
            with session_scope() as db:
                segment, project = _load(db, segment_id)
                synthesize_segment(db, get_storage(), segment, project)
        emit_segment_ready(project_id, segment_id)
        emit_stage_completed(project_id, STAGE)
        # Re-mux so the downloadable video carries the new line. mux_export
        # accepts a mix of muxed and freshly synthesized segments and sets the
        # project back to ready when it finishes.
        mux_export.delay(project_id)
        return segment_id
    except Exception as exc:  # noqa: BLE001
        _mark_project_failed(project_id, STAGE, exc)
        raise
    finally:
        clear_context()
