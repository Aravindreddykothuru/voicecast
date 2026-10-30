from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.export_job import ExportStatus


class ExportRequest(BaseModel):
    format: str = "mp4"
    resolution: str = "source"


class ExportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    status: ExportStatus
    format: str
    resolution: str
    output_url: str | None
    # The storage backend already knows this, and the client cannot ask for it
    # itself: a HEAD to output_url fails in Chrome whenever a <video> is
    # streaming the same URL (the media load's opaque response is reused and
    # the CORS check then fails), and cache-busting the HEAD would invalidate
    # an S3 presigned signature. None when the object is missing or the
    # backend cannot say.
    output_size_bytes: int | None = None
    qa_report: dict | None
    created_at: datetime
    completed_at: datetime | None
