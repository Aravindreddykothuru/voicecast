"""PATCH /api/segments/{id} and POST /api/segments/{id}/regenerate."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.routes_projects import require_idle
from app.db import get_db
from app.deps import get_owned_segment
from app.models.project import Project, ProjectStatus
from app.models.segment import Segment
from app.pipeline.regenerate import regenerate_segment as regenerate_segment_task
from app.schemas.common import OkResponse
from app.schemas.segment import RegenerateRequest, SegmentPatch, SegmentRead, segment_read

router = APIRouter(prefix="/api/segments", tags=["segments"])


@router.patch("/{segment_id}", response_model=SegmentRead)
def patch_segment(
    body: SegmentPatch,
    db: Session = Depends(get_db),
    segment: Segment = Depends(get_owned_segment),
):
    # An edit landing while a stage is working on this segment is silently
    # overwritten (translate) or silently ignored (synthesize already read
    # the old text). Refuse it instead.
    require_idle(db, db.get(Project, segment.project_id), "edit a segment")
    if body.translated_text is not None:
        segment.translated_text = body.translated_text
    if body.emotion_label is not None:
        segment.emotion_label = body.emotion_label
        segment.emotion_overridden = True
    db.commit()
    db.refresh(segment)
    # Same reason as list_segments: the audio columns are keys, the fields
    # are URLs.
    return segment_read(segment)


@router.post("/{segment_id}/regenerate", response_model=OkResponse, status_code=202)
def regenerate_segment(
    body: RegenerateRequest,
    db: Session = Depends(get_db),
    segment: Segment = Depends(get_owned_segment),
):
    """Queues a re-render of this segment only, then a re-mux of the export.

    The project is marked processing (current_stage "regenerate") until the
    re-mux finishes, so the UI can show that work is in flight, the dashboard
    download is not the stale video, and liveness applies to it like any run.
    Progress and failures arrive on the project's WebSocket channel.
    """
    project = db.get(Project, segment.project_id)
    if project.status == ProjectStatus.awaiting_language_confirmation:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Confirm the source language before regenerating segments.")
    require_idle(db, project, "regenerate a segment")
    project.status = ProjectStatus.processing
    project.current_stage = "regenerate"
    project.error_message = None
    project.error_is_permanent = None
    db.commit()
    regenerate_segment_task.delay(segment.id, body.stages)
    return OkResponse()
