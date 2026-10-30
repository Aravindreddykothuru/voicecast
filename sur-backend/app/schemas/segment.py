from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.segment import EmotionLabel, SegmentStatus


def segment_read(segment) -> "SegmentRead":
    """SegmentRead with its audio fields turned into URLs a client can fetch.

    The columns hold storage KEYS, not URLs, and the fields are named
    `..._url`. GET /export already presigned its key before returning it;
    the segment routes returned the raw row, so `tts_audio_url` arrived as
    "projects/<id>/segments/<id>/tts.wav". Resolved against the API base
    that is a 404, and the browser blocked it (ERR_BLOCKED_BY_ORB), so no
    segment audio ever played under STORAGE_BACKEND=local. With S3 it
    happened to work, because a presigned URL is absolute and the client
    left it alone -- which is why this stayed hidden.
    """
    from app.storage import get_storage

    read = SegmentRead.model_validate(segment, from_attributes=True)
    storage = get_storage()
    patch = {}
    for field in ("source_audio_url", "tts_audio_url"):
        key = getattr(read, field)
        if key and not key.startswith(("http://", "https://", "/")):
            patch[field] = storage.presigned_get_url(key)
    return read.model_copy(update=patch) if patch else read


class SegmentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    speaker_id: str | None
    index: int
    start_ms: int
    end_ms: int
    source_text: str | None
    detected_language: str | None
    detected_language_confidence: float | None
    translated_text: str | None
    emotion_label: EmotionLabel | None
    emotion_score: float | None
    emotion_overridden: bool
    source_audio_url: str | None
    tts_audio_url: str | None
    tts_duration_ms: int | None
    sync_offset_pct: float | None
    status: SegmentStatus
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class SegmentPatch(BaseModel):
    """PATCH /api/segments/{id} -- edit translated text or override emotion.

    All fields optional; only provided fields are updated. Editing either
    field does NOT itself trigger a re-render -- call /regenerate for that,
    so a reviewer can queue up several text edits before spending a TTS call.
    """

    translated_text: str | None = Field(default=None, max_length=4000)
    emotion_label: EmotionLabel | None = None


class RegenerateRequest(BaseModel):
    """POST /api/segments/{id}/regenerate.

    `stages` lets the caller ask for exactly what changed: an edited
    translation or an emotion override needs ["synthesize"] (re-translating
    would throw the edit away); ["translate", "synthesize"] re-derives the
    text too. The export is re-muxed afterwards either way.

    Validated, not filtered: unknown stage names used to be dropped silently
    (and an empty result fell back to the default), and ["translate"] alone
    left the rendered audio saying the old line.
    """

    stages: list[Literal["translate", "synthesize"]] = Field(
        default_factory=lambda: ["translate", "synthesize"], min_length=1
    )

    @field_validator("stages")
    @classmethod
    def _must_resynthesize(cls, stages: list[str]) -> list[str]:
        if "synthesize" not in stages:
            raise ValueError(
                "stages must include 'synthesize': re-translating without re-voicing "
                "leaves the dub saying the old line"
            )
        return sorted(set(stages), key=["translate", "synthesize"].index)
