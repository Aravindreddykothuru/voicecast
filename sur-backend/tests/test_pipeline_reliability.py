"""Tests for BUG 1 (permanent errors fail instead of getting stuck) and BUG 2 (worker health)."""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

from app.db import session_scope
from app.models.project import Project, ProjectStatus
from app.models.segment import Segment, SegmentStatus
from app.models.source_video import SourceVideo, SourceVideoStatus
from app.models.user import User
from app.pipeline import tasks as pipeline_tasks
from app.pipeline.maintenance import sweep_stuck_projects
from app.providers.base import TranscriptResult, SpeakerChunk


def _seed(fake_storage, *, review_language: bool = True):
    with session_scope() as db:
        user = User(email="dev@sur.local")
        db.add(user)
        db.flush()
        project = Project(
            user_id=user.id,
            title="ReliabilityTest",
            target_languages=["te"],
            review_language=review_language,
        )
        db.add(project)
        db.flush()
        video = SourceVideo(
            project_id=project.id,
            original_filename="test.mp4",
            storage_key="src.mp4",
            status=SourceVideoStatus.uploaded,
        )
        db.add(video)
        db.flush()
        ids = (project.id, video.id)

    local = pathlib.Path(tempfile.mkdtemp()) / "src.mp4"
    local.write_bytes(b"fake")
    fake_storage.upload_file("src.mp4", str(local))
    return ids


class _EmptySpeechASR:
    def transcribe_chunk(self, audio_path, start_ms, end_ms, language=None):
        return TranscriptResult(text="", language=None, confidence=0.0)


class _ValidASR:
    def transcribe_chunk(self, audio_path, start_ms, end_ms, language=None):
        return TranscriptResult(text="Hello world, this is a test.", language="en", confidence=0.95)


def test_empty_diarization_or_no_speech_fails_as_permanent_error(fake_storage):
    """BUG 1: Diarization / ASR with zero valid speech must end as failed with clear
    error message and current_stage set, and NEVER reach awaiting_language_confirmation.
    """
    project_id, video_id = _seed(fake_storage, review_language=True)

    with (
        patch("app.pipeline.tasks.ffmpeg_utils.extract_audio_wav", lambda s_, d: open(d, "wb").write(b"wav")),
        patch("app.pipeline.tasks.ffmpeg_utils.probe_duration_ms", lambda p: 5000),
        patch("app.pipeline.tasks.get_asr_provider", lambda: _EmptySpeechASR()),
    ):
        pipeline_tasks.extract_audio.apply(args=[project_id, video_id]).get()
        pipeline_tasks.chunk_and_diarize.apply(args=[project_id, video_id]).get()
        try:
            pipeline_tasks.transcribe.apply(args=[project_id]).get()
        except Exception:
            pass

    with session_scope() as db:
        project = db.get(Project, project_id)
        assert project.status == ProjectStatus.failed
        assert project.status != ProjectStatus.awaiting_language_confirmation
        assert project.current_stage == "transcribe"
        assert project.error_is_permanent is True
        assert "no speech" in (project.error_message or "").lower() or "no segments" in (project.error_message or "").lower()


def test_valid_diarization_reaches_awaiting_language_confirmation(fake_storage):
    """BUG 1: Valid diarization + transcription properly transitions to awaiting_language_confirmation."""
    project_id, video_id = _seed(fake_storage, review_language=True)

    with (
        patch("app.pipeline.tasks.ffmpeg_utils.extract_audio_wav", lambda s_, d: open(d, "wb").write(b"wav")),
        patch("app.pipeline.tasks.ffmpeg_utils.probe_duration_ms", lambda p: 5000),
        patch("app.pipeline.tasks.get_asr_provider", lambda: _ValidASR()),
    ):
        pipeline_tasks.extract_audio.apply(args=[project_id, video_id]).get()
        pipeline_tasks.chunk_and_diarize.apply(args=[project_id, video_id]).get()
        pipeline_tasks.transcribe.apply(args=[project_id]).get()

    with session_scope() as db:
        project = db.get(Project, project_id)
        assert project.status == ProjectStatus.awaiting_language_confirmation
        assert project.error_message is None


def test_sweep_stuck_projects_cleans_up_orphaned_awaiting(fake_storage):
    """BUG 1: Periodic/startup sweep finds projects stuck in awaiting_language_confirmation
    with 0 segments and marks them failed.
    """
    project_id, _ = _seed(fake_storage, review_language=True)

    with session_scope() as db:
        p = db.get(Project, project_id)
        p.status = ProjectStatus.awaiting_language_confirmation

    cleaned = sweep_stuck_projects()
    assert cleaned >= 1

    with session_scope() as db:
        p = db.get(Project, project_id)
        assert p.status == ProjectStatus.failed
        assert p.current_stage == "transcribe"
        assert p.error_is_permanent is True
        assert "no segments" in (p.error_message or "").lower()


def test_worker_health_endpoint_contract(client):
    """BUG 2: GET /api/health/workers returns workers_online, queues dict, and oldest_queued_age_seconds."""
    resp = client.get("/api/health/workers")
    assert resp.status_code == 200
    data = resp.json()
    assert "workers_online" in data
    assert isinstance(data["workers_online"], int)
    assert "queues" in data
    assert isinstance(data["queues"], dict)
    assert "q.extract_audio" in data["queues"]
    assert "q.translate" in data["queues"]
    assert "oldest_queued_age_seconds" in data


def test_unpronounceable_text_segment_synthesizes_as_silent(fake_storage):
    """Punctuation-only or unpronounceable translation (e.g. '.....') leaves segment silent without failing."""
    from app.pipeline.tasks import synthesize_segment
    project_id, video_id = _seed(fake_storage, review_language=False)

    with session_scope() as db:
        project = db.get(Project, project_id)
        seg = Segment(
            project_id=project_id,
            source_video_id=video_id,
            index=0,
            start_ms=0,
            end_ms=2000,
            source_text="...",
            translated_text="............................................................",
            status=SegmentStatus.translated,
        )
        db.add(seg)
        db.commit()
        db.refresh(seg)
        db.refresh(project)

        synthesize_segment(db, fake_storage, seg, project)
        assert seg.status == SegmentStatus.synthesized
        assert seg.tts_audio_url is None
        assert seg.tts_duration_ms is None

