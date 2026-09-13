from __future__ import annotations

from app.db import session_scope
from app.models.project import Project
from app.models.segment import EmotionLabel, Segment, SegmentStatus
from app.models.source_video import SourceVideo, SourceVideoStatus
from app.models.user import User


def _make_segment(fake_storage) -> tuple[str, str]:
    with session_scope() as db:
        user = User(email="dev@sur.local")
        db.add(user)
        db.flush()
        project = Project(user_id=user.id, title="F", target_languages=["te"])
        db.add(project)
        db.flush()
        video = SourceVideo(
            project_id=project.id,
            original_filename="m.mp4",
            storage_key="k",
            audio_storage_key="k/audio.wav",
            duration_ms=2000,  # set by extract_audio; mux_export only probes when missing
            status=SourceVideoStatus.extracted,
        )
        db.add(video)
        db.flush()
        segment = Segment(
            project_id=project.id,
            source_video_id=video.id,
            index=0,
            start_ms=0,
            end_ms=2000,
            source_text="Get out!",
            detected_language="en",  # regenerate("translate") now requires this
            detected_language_confidence=0.95,
            translated_text="[TE] Get out!",
            emotion_label=EmotionLabel.anger,
            emotion_score=0.9,
            status=SegmentStatus.translated,
        )
        db.add(segment)
        db.flush()
        return project.id, segment.id


def test_patch_segment_updates_text_without_triggering_render(client, fake_storage):
    _, segment_id = _make_segment(fake_storage)
    resp = client.patch(f"/api/segments/{segment_id}", json={"translated_text": "edited text"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["translated_text"] == "edited text"
    # PATCH alone shouldn't have run a TTS render.
    assert body["status"] == "translated"


def test_patch_segment_emotion_override_sets_flag(client, fake_storage):
    _, segment_id = _make_segment(fake_storage)
    resp = client.patch(f"/api/segments/{segment_id}", json={"emotion_label": "happiness"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["emotion_label"] == "happiness"
    assert body["emotion_overridden"] is True


def test_regenerate_segment_rerenders_it_and_re_exports_the_video(client, fake_storage):
    """A regenerated line used to be audible in the editor while the
    downloadable video still said the old one: nothing re-muxed."""
    from unittest.mock import patch

    import pathlib
    import tempfile

    from app.models.export_job import ExportJob
    from app.models.project import ProjectStatus
    from sqlalchemy import select

    project_id, segment_id = _make_segment(fake_storage)
    local = pathlib.Path(tempfile.mkdtemp()) / "src.mp4"
    local.write_bytes(b"fake")
    fake_storage.upload_file("k", str(local))

    muxed = []
    with patch("app.pipeline.tasks.ffmpeg_utils.mux_timeline",
               lambda v, placements, out: (muxed.append(len(placements)), open(out, "wb").write(b"mp4"))):
        resp = client.post(f"/api/segments/{segment_id}/regenerate", json={"stages": ["translate", "synthesize"]})
    assert resp.status_code == 202

    with session_scope() as db:
        segment = db.get(Segment, segment_id)
        assert segment.tts_audio_url
        assert db.get(Project, project_id).status == ProjectStatus.ready
        assert db.execute(select(ExportJob).where(ExportJob.project_id == project_id)).scalars().first().output_url
    assert muxed == [1]


def test_regenerate_missing_segment_task_raises():
    from app.pipeline.regenerate import regenerate_segment

    try:
        regenerate_segment.apply(args=["does-not-exist", ["translate"]]).get()
        raise AssertionError("expected failure for missing segment")
    except ValueError:
        pass
