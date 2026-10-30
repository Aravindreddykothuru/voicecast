from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.project import ProjectStatus


class ProjectCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    target_languages: list[str] = Field(..., min_length=1)


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    target_languages: list[str]
    status: ProjectStatus
    current_stage: str | None
    preserve_emotion: bool
    clone_voice: bool
    lip_sync_aware: bool
    tts_model: str | None
    error_message: str | None
    error_is_permanent: bool | None = None
    source_language: str | None = None
    review_language: bool = True
    created_at: datetime
    updated_at: datetime
    # Liveness, computed by the backend (app/pipeline/liveness.py). A run
    # nobody is working on used to look identical to a slow one.
    last_activity_at: datetime | None = None
    stalled: bool = False
    stalled_reason: str | None = None
    # The source language the backend will use if the language gate is
    # accepted as-is: the majority of ASR detections over segments with
    # speech, and the mean confidence of those votes. The UI shows THIS
    # instead of recounting segments itself -- its own count included silent
    # segments and could name a different language than the one accepted.
    detected_source_language: str | None = None
    detected_source_language_confidence: float | None = None


class ProjectListItem(ProjectRead):
    """Adds the fields the Dashboard screen's project cards need."""

    source_video_duration_ms: int | None = None
    segment_count: int = 0


class UploadUrlRequest(BaseModel):
    filename: str
    content_type: str = "video/mp4"


class UploadUrlResponse(BaseModel):
    source_video_id: str
    upload_url: str
    storage_key: str
    method: str = "PUT"
    expires_in: int = 3600


class UploadConfirmRequest(BaseModel):
    source_video_id: str
    duration_ms: int | None = None


class ConfirmLanguageRequest(BaseModel):
    """POST /api/projects/{id}/confirm-language.

    `source_language` overrides what ASR detected. None accepts the detection
    as-is. Either way the pipeline resumes from emotion detection onward.
    """

    source_language: str | None = None
    # Re-run ASR forced to the chosen (or detected) language even when it
    # matches the detection. Detection runs per chunk, so a mostly-correct
    # run can still have misdetected lines that forcing the language fixes.
    # Without this flag "re-run ASR" with the detected language silently
    # did nothing.
    force_retranscribe: bool = False


class ProcessRequest(BaseModel):
    """Body for POST /api/projects/{id}/process -- section 05 of the PRD."""

    preserve_emotion: bool = True
    clone_voice: bool = False
    lip_sync_aware: bool = False
    tts_model: str | None = None
    # Pause after ASR so the detected source language can be confirmed before
    # the expensive stages run. Defaults to on: silently dubbing from a
    # mis-detected language is worse than one extra click.
    review_language: bool = True
    # The source language, when the caller already knows it. ASR then decodes
    # in that language instead of detecting, and there is nothing left to
    # review, so review_language is ignored. The UI used to "pin" a language
    # by calling /confirm-language right after /process -- which 409s,
    # because a run that isn't reviewing never parks at the gate, so the
    # choice was silently dropped and ASR auto-detected anyway.
    source_language: str | None = None
