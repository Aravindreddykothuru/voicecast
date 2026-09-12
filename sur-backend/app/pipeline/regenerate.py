"""Single-segment regeneration -- POST /api/segments/{id}/regenerate.

Deliberately NOT a Celery chain of the full-project tasks: those operate on
"all segments in status X" and would either skip this one segment (if its
status doesn't match) or require resetting its status first (racy against
whatever else is touching the project). Instead this runs the requested
stages against exactly one segment, reusing the same provider-calling
helpers the full-project tasks use, so behavior never drifts between the two
paths.

Split across two queues, exactly like the stages it re-runs. It used to be a
single task on q.synthesize that also translated -- but the q.synthesize
worker runs in the TTS venv, which has no IndicTrans2 (the two stacks pin
incompatible torch/transformers; see requirements-tts.txt), so every
"translate" regenerate failed on import in production. Now:

  regenerate_segment    q.translate   validate, translate if asked, then hand off
  regenerate_synthesize q.synthesize  render if asked
"""
from __future__ import annotations

import logging

from app.celery_app import QUEUE_TRANSLATE, QUEUE_TTS, celery_app
from app.db import session_scope
from app.logging_conf import bind_context, clear_context
from app.models.base import is_uuid
from app.models.project import Project
from app.models.segment import Segment
from app.pipeline.events import emit_error, emit_segment_ready, emit_stage_completed, emit_stage_started
from app.pipeline.tasks import has_speech, synthesize_segment, translate_segment
from app.storage import get_storage

logger = logging.getLogger(__name__)

VALID_STAGES = {"translate", "synthesize"}
_PERMANENT = (ValueError, LookupError, TypeError)


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
    requested = [s for s in stages if s in VALID_STAGES] or ["translate", "synthesize"]

    with session_scope() as db:
        _, project = _load(db, segment_id)
        project_id = project.id

    bind_context(project_id=project_id, segment_id=segment_id)
    emit_stage_started(project_id, "regenerate")
    try:
        if "translate" in requested:
            with session_scope() as db:
                segment, project = _load(db, segment_id)
                if not project.target_languages:
                    raise ValueError(f"project {project.id} has no target_languages configured")
                # Same rule as the full-project translate stage: never guess
                # a source language. Confirmed project.source_language wins;
                # this segment's own ASR detection is the fallback.
                source_lang = project.source_language or segment.detected_language
                if not source_lang and has_speech(segment):
                    raise ValueError(
                        f"segment {segment.id} has no confirmed or detected source "
                        "language -- cannot translate without guessing."
                    )
                translate_segment(segment, project.target_languages[0], source_lang or "")
            emit_segment_ready(project_id, segment_id)

        if "synthesize" in requested:
            # Its own queue, its own worker; see the module docstring.
            regenerate_synthesize.delay(segment_id)
        else:
            emit_stage_completed(project_id, "regenerate")
        return segment_id
    except Exception as exc:  # noqa: BLE001
        emit_error(project_id, "regenerate", str(exc), permanent=isinstance(exc, _PERMANENT))
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
        with session_scope() as db:
            segment, project = _load(db, segment_id)
            synthesize_segment(db, get_storage(), segment, project)
        emit_segment_ready(project_id, segment_id)
        emit_stage_completed(project_id, "regenerate")
        return segment_id
    except Exception as exc:  # noqa: BLE001
        emit_error(project_id, "regenerate", str(exc), permanent=isinstance(exc, _PERMANENT))
        raise
    finally:
        clear_context()
