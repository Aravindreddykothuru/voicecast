from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.segment import EmotionLabel, SegmentStatus


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
