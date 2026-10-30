"""Run lifecycle contracts found by auditing the frontend against the backend.

Every test names a user-visible failure: a Retry button that killed the run,
a second click that started a duplicate pipeline, a spinner that never ended,
a "re-run ASR" button that did nothing, a regenerate that never reached the
download, and a TTS language offered on weights the business can't ship.
"""
from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import patch

import pytest
from sqlalchemy import select, text

from app.db import session_scope
from app.models.project import Project, ProjectStatus
from app.models.segment import Segment, SegmentStatus
from app.models.source_video import SourceVideo, SourceVideoStatus
from app.models.user import User
from app.pipeline import tasks as pipeline_tasks


def _seed(fake_storage, *, status=ProjectStatus.draft, seg_status=SegmentStatus.translated, detected=("en", "en", "hi")):
    local = pathlib.Path(tempfile.mkdtemp()) / "src.mp4"
    local.write_bytes(b"fake")
    with session_scope() as db:
        user = User(email="dev@sur.local")
        db.add(user)
        db.flush()
        project = Project(user_id=user.id, title="L", target_languages=["te"], status=status, source_language=None)
        db.add(project)
        db.flush()
        video = SourceVideo(project_id=project.id, original_filename="m.mp4", storage_key=f"{project.id}/src.mp4",
                            audio_storage_key=f"{project.id}/audio.wav", duration_ms=12000,
                            status=SourceVideoStatus.extracted)
        db.add(video)
        db.flush()
        fake_storage.upload_file(video.storage_key, str(local))
        fake_storage.upload_file(video.audio_storage_key, str(local))
        seg_ids = []
        for i, lang in enumerate(detected):
            s = Segment(project_id=project.id, source_video_id=video.id, index=i, start_ms=i * 3000,
                        end_ms=i * 3000 + 2000, source_text="" if lang is None else f"line {i}",
                        detected_language=lang, detected_language_confidence=0.9, status=seg_status,
                        translated_text=f"[TE] line {i}")
            db.add(s)
            db.flush()
            seg_ids.append(s.id)
        return project.id, seg_ids


def _age(project_id, *, minutes=60, heartbeat_minutes=None):
    with session_scope() as db:
        hb = "NULL" if heartbeat_minutes is None else f"now() - interval '{heartbeat_minutes} minutes'"
        db.execute(text(f"UPDATE projects SET updated_at = now() - interval '{minutes} minutes', heartbeat_at = {hb} WHERE id = :id"),
                   {"id": project_id})
        db.execute(text(f"UPDATE segments SET updated_at = now() - interval '{minutes} minutes' WHERE project_id = :id"),
                   {"id": project_id})


# --- resume ------------------------------------------------------------------
def test_restarting_a_run_that_failed_late_resumes_instead_of_dying_at_transcribe(fake_storage):
    """The dashboard's Retry re-posts /process, which re-enqueues the chain
    from the start. transcribe found no pending segment (all were already
    translated) and failed the project permanently with "no segments"."""
    project_id, _ = _seed(fake_storage, seg_status=SegmentStatus.translated)
    pipeline_tasks.transcribe.apply(args=[project_id]).get()
    with session_scope() as db:
        assert db.get(Project, project_id).status != ProjectStatus.failed


def test_plain_transcribe_does_not_redo_finished_segments_but_retranscribe_does(fake_storage):
    project_id, seg_ids = _seed(fake_storage, seg_status=SegmentStatus.transcribed)
    calls = []

    class Spy:
        def transcribe_chunk(self, audio_path, start_ms, end_ms, language=None):
            from app.providers.base import TranscriptResult

            calls.append(start_ms)
            return TranscriptResult(text="again", confidence=0.9, language="en")

    with patch("app.pipeline.tasks.get_asr_provider", lambda: Spy()):
        pipeline_tasks.transcribe.apply(args=[project_id]).get()
        assert calls == []
        pipeline_tasks.transcribe.apply(args=[project_id], kwargs={"retranscribe": True}).get()
    assert len(calls) == len(seg_ids)


# --- liveness ------------------------------------------------------------------
def test_a_queued_run_no_worker_picked_up_is_reported_stalled(client, fake_storage):
    project_id, _ = _seed(fake_storage, status=ProjectStatus.queued)
    body = client.get(f"/api/projects/{project_id}").json()
    assert body["stalled"] is False

    _age(project_id)
    body = client.get(f"/api/projects/{project_id}").json()
    assert body["stalled"] is True
    assert "No worker has picked this run up" in body["stalled_reason"]
    assert client.get("/api/projects").json()[0]["stalled"] is True


def test_a_processing_run_whose_worker_went_silent_is_reported_stalled(client, fake_storage):
    project_id, _ = _seed(fake_storage, status=ProjectStatus.processing)
    with session_scope() as db:
        db.get(Project, project_id).current_stage = "synthesize"
    _age(project_id, minutes=60, heartbeat_minutes=30)
    body = client.get(f"/api/projects/{project_id}").json()
    assert body["stalled"] is True
    assert "synthesize stopped reporting" in body["stalled_reason"]


def test_a_live_heartbeat_keeps_a_long_stage_from_looking_stalled(client, fake_storage):
    project_id, _ = _seed(fake_storage, status=ProjectStatus.processing)
    _age(project_id, minutes=60, heartbeat_minutes=0)
    assert client.get(f"/api/projects/{project_id}").json()["stalled"] is False


def test_pipeline_tasks_heartbeat_while_they_run(fake_storage):
    project_id, _ = _seed(fake_storage, seg_status=SegmentStatus.translated)
    pipeline_tasks.transcribe.apply(args=[project_id]).get()
    with session_scope() as db:
        assert db.get(Project, project_id).heartbeat_at is not None


def test_the_heartbeat_thread_stops_when_the_task_ends(fake_storage, monkeypatch):
    import threading

    from app.config import get_settings
    from app.pipeline.liveness import heartbeat

    monkeypatch.setattr(get_settings(), "heartbeat_interval_seconds", 1)
    project_id, _ = _seed(fake_storage)
    before = {t.name for t in threading.enumerate()}
    with heartbeat(project_id):
        assert any(t.name.startswith("heartbeat-") for t in threading.enumerate())
    assert {t.name for t in threading.enumerate() if t.name.startswith("heartbeat-")} <= before


# --- guards ----------------------------------------------------------------------
def test_process_refuses_to_start_a_second_pipeline_on_a_live_run(client, fake_storage):
    project_id, _ = _seed(fake_storage, status=ProjectStatus.processing)
    with patch("app.api.routes_projects.start_pipeline") as start:
        resp = client.post(f"/api/projects/{project_id}/process", json={})
    assert resp.status_code == 409
    start.assert_not_called()


def test_process_may_restart_a_stalled_run(client, fake_storage):
    project_id, _ = _seed(fake_storage, status=ProjectStatus.processing)
    _age(project_id)
    with patch("app.api.routes_projects.start_pipeline") as start:
        resp = client.post(f"/api/projects/{project_id}/process", json={"review_language": False})
    assert resp.status_code == 200, resp.text
    start.assert_called_once()
    assert resp.json()["status"] == "queued"


def test_process_refuses_a_run_waiting_at_the_language_gate(client, fake_storage):
    project_id, _ = _seed(fake_storage, status=ProjectStatus.awaiting_language_confirmation)
    with patch("app.api.routes_projects.start_pipeline") as start:
        resp = client.post(f"/api/projects/{project_id}/process", json={})
    assert resp.status_code == 409 and "confirm-language" in resp.json()["detail"]
    start.assert_not_called()


def test_segment_edits_are_refused_while_a_run_is_live(client, fake_storage):
    project_id, seg_ids = _seed(fake_storage, status=ProjectStatus.processing)
    assert client.patch(f"/api/segments/{seg_ids[0]}", json={"translated_text": "x"}).status_code == 409
    assert client.post(f"/api/segments/{seg_ids[0]}/regenerate", json={"stages": ["synthesize"]}).status_code == 409


# --- language gate ---------------------------------------------------------------
def test_gate_reports_the_same_detection_it_will_accept(client, fake_storage):
    project_id, _ = _seed(fake_storage, status=ProjectStatus.awaiting_language_confirmation,
                          seg_status=SegmentStatus.transcribed, detected=("hi", "en", "en", None, None))
    body = client.get(f"/api/projects/{project_id}").json()
    assert body["detected_source_language"] == "en"
    assert body["detected_source_language_confidence"] == pytest.approx(0.9)
    with patch("app.api.routes_projects.continue_pipeline"):
        confirmed = client.post(f"/api/projects/{project_id}/confirm-language", json={}).json()
    assert confirmed["source_language"] == "en"


def test_force_retranscribe_reruns_asr_even_with_the_detected_language(client, fake_storage):
    """"Re-run ASR with this language" sent the detected code, which the
    backend treated as a plain accept -- the button did nothing."""
    project_id, _ = _seed(fake_storage, status=ProjectStatus.awaiting_language_confirmation,
                          seg_status=SegmentStatus.transcribed, detected=("en", "en"))
    with patch("app.api.routes_projects.start_pipeline") as start, patch("app.api.routes_projects.continue_pipeline") as cont:
        resp = client.post(f"/api/projects/{project_id}/confirm-language",
                           json={"source_language": "en", "force_retranscribe": True})
    assert resp.status_code == 200, resp.text
    cont.assert_not_called()
    assert start.call_args.kwargs == {"review_language": False, "retranscribe": True}


def test_gate_rejects_an_unsupported_override(client, fake_storage):
    project_id, _ = _seed(fake_storage, status=ProjectStatus.awaiting_language_confirmation,
                          seg_status=SegmentStatus.transcribed)
    resp = client.post(f"/api/projects/{project_id}/confirm-language", json={"source_language": "zh"})
    assert resp.status_code == 422


# --- regenerate --------------------------------------------------------------------
@pytest.mark.parametrize("stages", [["translate"], [], ["transcribe", "synthesize"], ["bogus"]])
def test_regenerate_rejects_stage_lists_that_would_do_nothing_or_leave_the_dub_stale(client, fake_storage, stages):
    project_id, seg_ids = _seed(fake_storage, status=ProjectStatus.ready)
    resp = client.post(f"/api/segments/{seg_ids[0]}/regenerate", json={"stages": stages})
    assert resp.status_code == 422, resp.text


def test_a_failed_regenerate_marks_the_project_failed_instead_of_stranding_it(client, fake_storage):
    project_id, seg_ids = _seed(fake_storage, status=ProjectStatus.ready)

    class Broken:
        def synthesize(self, request):
            raise RuntimeError("tts exploded")

    with patch("app.pipeline.tasks.get_tts_provider", lambda: Broken()):
        with pytest.raises(Exception):
            client.post(f"/api/segments/{seg_ids[0]}/regenerate", json={"stages": ["synthesize"]})
    with session_scope() as db:
        project = db.get(Project, project_id)
        assert project.status == ProjectStatus.failed
        assert "tts exploded" in project.error_message


# --- licensing ---------------------------------------------------------------------
@pytest.fixture
def real_tts(monkeypatch):
    from app.config import get_settings
    from app.providers import registry

    s = get_settings()
    monkeypatch.setattr(s, "tts_provider", "real")
    registry.get_tts_provider.cache_clear()
    yield s
    registry.get_tts_provider.cache_clear()


def test_commercial_deployments_only_offer_languages_with_a_commercially_licensed_voice(client, real_tts, monkeypatch):
    monkeypatch.setattr(real_tts, "tts_engine", "syspin")
    monkeypatch.setattr(real_tts, "tts_require_commercial_license", True)
    caps = client.get("/api/capabilities").json()
    available = sorted(l["code"] for l in caps["languages"] if l["tts_available"])
    assert available == ["bn", "hi", "kn", "mr", "te"]
    assert caps["tts_engine"] == "syspin" and caps["tts_licenses"] == ["CC-BY-4.0"] and caps["tts_commercial_use"] is True

    resp = client.post("/api/projects", json={"title": "T", "target_languages": ["ta"]})
    assert resp.status_code == 422 and "no text-to-speech voice" in resp.json()["detail"]


def test_a_non_commercial_engine_is_reported_as_such(client, real_tts, monkeypatch):
    monkeypatch.setattr(real_tts, "tts_engine", "mms")
    monkeypatch.setattr(real_tts, "tts_require_commercial_license", False)
    caps = client.get("/api/capabilities").json()
    assert all(l["tts_available"] for l in caps["languages"])
    assert caps["tts_licenses"] == ["CC-BY-NC-4.0"] and caps["tts_commercial_use"] is False


def test_the_registry_refuses_a_non_commercial_engine_when_commercial_use_is_required(real_tts, monkeypatch):
    from app.providers import registry

    monkeypatch.setattr(real_tts, "tts_engine", "mms")
    monkeypatch.setattr(real_tts, "tts_require_commercial_license", True)
    with pytest.raises(RuntimeError, match="non-commercially licensed"):
        registry.get_tts_provider()
