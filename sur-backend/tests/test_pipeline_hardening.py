"""Regression tests for failures found running the real pipeline end to end.

Each test names the failure it guards. Mock providers and patched ffmpeg,
like the rest of the suite; the real-media side of the same fixes is in
tests/test_timeline_accuracy.py.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.db import session_scope
from app.models.export_job import ExportJob
from app.models.project import Project, ProjectStatus
from app.models.segment import EmotionLabel, Segment, SegmentStatus
from app.models.source_video import SourceVideo, SourceVideoStatus
from app.models.user import User
from app.pipeline import tasks as pipeline_tasks


def _project(fake_storage, *, texts, status=SegmentStatus.emotion_detected, **project_kwargs):
    import pathlib
    import tempfile

    local = pathlib.Path(tempfile.mkdtemp()) / "src.mp4"
    local.write_bytes(b"fake video")
    with session_scope() as db:
        user = User(email="dev@sur.local")
        db.add(user)
        db.flush()
        project = Project(user_id=user.id, title="P", target_languages=["te"], **project_kwargs)
        db.add(project)
        db.flush()
        video = SourceVideo(
            project_id=project.id, original_filename="m.mp4", storage_key=f"{project.id}/src.mp4",
            audio_storage_key=f"{project.id}/audio.wav", duration_ms=12000,
            status=SourceVideoStatus.extracted,
        )
        db.add(video)
        db.flush()
        fake_storage.upload_file(video.storage_key, str(local))
        fake_storage.upload_file(video.audio_storage_key, str(local))
        for i, text in enumerate(texts):
            db.add(Segment(
                project_id=project.id, source_video_id=video.id, index=i,
                start_ms=i * 3000, end_ms=i * 3000 + 2500, source_text=text,
                detected_language="en", detected_language_confidence=0.9,
                emotion_label=EmotionLabel.neutral if text.strip() else None,
                emotion_score=0.9 if text.strip() else None, status=status,
            ))
        return project.id


def _capture_mux():
    calls = []

    def fake(video, placements, out):
        calls.append(list(placements))
        with open(out, "wb") as f:
            f.write(b"mp4")

    return calls, fake


# --- segments with no speech ----------------------------------------------
def test_segments_without_speech_are_left_silent_not_translated_or_spoken(fake_storage):
    """ASR returns "" for a laugh or a VAD false positive. That used to be
    translated and handed to TTS, which either crashed or invented a line."""
    project_id = _project(fake_storage, texts=["Hello there.", "   ", "Goodbye."])
    calls, fake_mux = _capture_mux()
    translated = []

    class Spy:
        def translate(self, text, target_lang, src_lang="en"):
            translated.append(text)
            from app.providers.base import TranslationResult

            return TranslationResult(text=f"[TE] {text}", src_lang=src_lang, target_lang=target_lang)

    with (
        patch("app.pipeline.tasks.get_translation_provider", lambda: Spy()),
        patch("app.pipeline.tasks.ffmpeg_utils.mux_timeline", fake_mux),
    ):
        pipeline_tasks.translate.apply(args=[project_id]).get()
        pipeline_tasks.synthesize.apply(args=[project_id]).get()
        pipeline_tasks.mux_export.apply(args=[project_id]).get()

    assert translated == ["Hello there.", "Goodbye."]
    with session_scope() as db:
        segs = db.execute(select(Segment).where(Segment.project_id == project_id).order_by(Segment.index)).scalars().all()
        assert segs[1].translated_text == "" and segs[1].tts_audio_url is None
        assert segs[0].tts_audio_url and segs[2].tts_audio_url
        assert db.get(Project, project_id).status == ProjectStatus.ready
        report = db.execute(select(ExportJob).where(ExportJob.project_id == project_id)).scalar_one().qa_report
    assert [p.start_ms for p in calls[0]] == [0, 6000], "only spoken segments are placed, at their own timecodes"
    assert report["overall"]["segment_count"] == 3
    assert report["overall"]["speech_segment_count"] == 2
    assert report["segments"][1]["has_speech"] is False


def test_a_project_with_no_speech_anywhere_fails_permanently_with_a_reason(fake_storage):
    project_id = _project(fake_storage, texts=["", " "])
    with pytest.raises(ValueError, match="no speech"):
        pipeline_tasks.translate.apply(args=[project_id]).get()
    with session_scope() as db:
        project = db.get(Project, project_id)
        assert project.status == ProjectStatus.failed and project.error_is_permanent is True


# --- per-segment commits ----------------------------------------------------
def test_a_synthesize_failure_keeps_segments_already_rendered(fake_storage):
    """synthesize held every segment in one transaction: a failure on the
    last segment rolled back every finished render (minutes each on CPU)."""
    project_id = _project(fake_storage, texts=["one", "two", "three"], status=SegmentStatus.translated)
    with session_scope() as db:
        for s in db.execute(select(Segment).where(Segment.project_id == project_id)).scalars():
            s.translated_text = f"[TE] {s.source_text}"

    from app.providers.tts.mock_provider import MockTTSProvider

    rendered = []

    class FailsOnThird(MockTTSProvider):
        def synthesize(self, request):
            if request.text.endswith("three"):
                raise RuntimeError("worker lost")
            rendered.append(request.text)
            return super().synthesize(request)

    with patch("app.pipeline.tasks.get_tts_provider", lambda: FailsOnThird()):
        with pytest.raises(Exception):
            pipeline_tasks.synthesize.apply(args=[project_id]).get()

    with session_scope() as db:
        statuses = [s.status for s in db.execute(
            select(Segment).where(Segment.project_id == project_id).order_by(Segment.index)).scalars()]
    assert statuses == [SegmentStatus.synthesized, SegmentStatus.synthesized, SegmentStatus.translated]

    rendered.clear()
    with patch("app.pipeline.tasks.get_tts_provider", lambda: MockTTSProvider()):
        pipeline_tasks.synthesize.apply(args=[project_id]).get()
    with session_scope() as db:
        assert all(s.status == SegmentStatus.synthesized for s in db.execute(
            select(Segment).where(Segment.project_id == project_id)).scalars())


def test_tts_is_given_the_time_until_the_next_spoken_line_not_just_its_own_segment(fake_storage):
    """TTS used to be told to fit each line into its own segment length, so it
    sped up lines that had a pause after them the mux would have used. In the
    real run the two squeezed short lines were the two Whisper could not
    read back (CER 0.64 and 0.50)."""
    from app.pipeline.timeline import GUARD_MS
    from app.providers.tts.mock_provider import MockTTSProvider

    project_id = _project(fake_storage, texts=["one", " ", "three"], status=SegmentStatus.translated)
    with session_scope() as db:
        for s in db.execute(select(Segment).where(Segment.project_id == project_id)).scalars():
            s.translated_text = f"[TE] {s.source_text}" if s.source_text.strip() else ""

    captured = []

    class Spy(MockTTSProvider):
        def synthesize(self, request):
            captured.append((request.text, request.target_duration_ms))
            return super().synthesize(request)

    with patch("app.pipeline.tasks.get_tts_provider", lambda: Spy()):
        pipeline_tasks.synthesize.apply(args=[project_id]).get()

    # Segments at 0, 3000 (no speech) and 6000ms; the video is 12000ms long.
    assert captured == [("[TE] one", 6000 - GUARD_MS), ("[TE] three", 12000 - 6000 - GUARD_MS)]


def test_voice_cloning_uses_one_reference_per_speaker_built_from_their_longest_lines(fake_storage, monkeypatch):
    """Each segment used its own audio slice as the voice reference: 0.5s for
    a one-word line, and a different reference for every line. The cloned
    dub read back at corpus CER 0.30 against 0.16 unconverted."""
    import contextlib
    import os
    import tempfile

    from app.models.speaker import Speaker
    from app.providers.tts.mock_provider import MockTTSProvider

    project_id = _project(fake_storage, texts=["a", "b", "c"], status=SegmentStatus.translated, clone_voice=True)
    with session_scope() as db:
        speaker = Speaker(project_id=project_id, label="Speaker 1", diarization_tag="SPEAKER_00")
        db.add(speaker)
        db.flush()
        segs = db.execute(select(Segment).where(Segment.project_id == project_id).order_by(Segment.index)).scalars().all()
        for s, span in zip(segs, (1000, 2500, 2000)):
            s.speaker_id = speaker.id
            s.end_ms = s.start_ms + span
            s.translated_text = f"[TE] {s.source_text}"
        speaker_id = speaker.id

    sliced, joined, references = [], [], []

    @contextlib.contextmanager
    def fake_slice(audio_path, start_ms, end_ms):
        sliced.append((start_ms, end_ms))
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        yield path
        os.remove(path)

    def fake_concat(paths, out, sample_rate=16000):
        joined.append(len(paths))
        open(out, "wb").write(b"ref")

    class Spy(MockTTSProvider):
        def synthesize(self, request):
            references.append(request.voice_reference_path and os.path.exists(request.voice_reference_path))
            return super().synthesize(request)

    monkeypatch.setattr(pipeline_tasks, "REFERENCE_TARGET_MS", 3000)
    with (
        patch("app.pipeline.tasks.ffmpeg_utils.extract_audio_slice", fake_slice),
        patch("app.pipeline.tasks.ffmpeg_utils.concat_audio", fake_concat),
        patch("app.pipeline.tasks.get_tts_provider", lambda: Spy()),
    ):
        pipeline_tasks.synthesize.apply(args=[project_id]).get()

    # Longest first until the target: the 2.5s and 2.0s lines, in time order,
    # extracted once for the speaker -- not once per segment.
    assert sliced == [(3000, 5500), (6000, 8000)]
    assert joined == [2]
    assert references == [True, True, True]
    with session_scope() as db:
        assert db.get(Speaker, speaker_id).reference_clip_url == f"projects/{project_id}/speakers/{speaker_id}/reference.wav"


# --- timeline fitting reaches the mux --------------------------------------
def test_mux_speeds_up_a_clip_that_would_run_over_the_next_line(fake_storage):
    project_id = _project(fake_storage, texts=["a", "b"], status=SegmentStatus.synthesized)
    with session_scope() as db:
        segs = db.execute(select(Segment).where(Segment.project_id == project_id).order_by(Segment.index)).scalars().all()
        import pathlib
        import tempfile

        clip = pathlib.Path(tempfile.mkdtemp()) / "c.wav"
        clip.write_bytes(b"wav")
        for s, dur in zip(segs, (3600, 1000)):
            s.tts_audio_url = f"{s.id}.wav"
            s.tts_duration_ms = dur
            fake_storage.upload_file(s.tts_audio_url, str(clip))
    calls, fake_mux = _capture_mux()
    with patch("app.pipeline.tasks.ffmpeg_utils.mux_timeline", fake_mux):
        pipeline_tasks.mux_export.apply(args=[project_id]).get()
    first, second = calls[0]
    assert first.start_ms == 0 and first.tempo > 1.0, "3.6s clip with 3s before the next line must be fitted"
    assert second.tempo == 1.0


# --- /process contract -----------------------------------------------------
def _uploaded_project(client, fake_storage):
    resp = client.post("/api/projects", json={"title": "T", "target_languages": ["te"]})
    project_id = resp.json()["id"]
    with session_scope() as db:
        db.add(SourceVideo(project_id=project_id, original_filename="m.mp4", storage_key="k",
                           status=SourceVideoStatus.uploaded))
    return project_id


def test_process_passes_review_language_through_to_the_chain(client, fake_storage):
    """/process omitted review_language, so start_pipeline's default (True)
    queued only up to transcribe while transcribe itself did not park: the
    project sat in "processing" forever with nothing queued."""
    project_id = _uploaded_project(client, fake_storage)
    with patch("app.api.routes_projects.start_pipeline") as start:
        resp = client.post(f"/api/projects/{project_id}/process", json={"review_language": False})
    assert resp.status_code == 200, resp.text
    assert start.call_args.kwargs["review_language"] is False


def test_process_accepts_a_known_source_language_and_skips_the_gate(client, fake_storage):
    """The UI used to pin a language by calling /confirm-language right after
    /process -- a 409, so the choice was silently dropped."""
    project_id = _uploaded_project(client, fake_storage)
    with patch("app.api.routes_projects.start_pipeline") as start:
        resp = client.post(f"/api/projects/{project_id}/process",
                           json={"review_language": True, "source_language": "hi"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["source_language"] == "hi"
    assert resp.json()["review_language"] is False
    assert start.call_args.kwargs["review_language"] is False


def test_process_rejects_an_unsupported_source_language(client, fake_storage):
    project_id = _uploaded_project(client, fake_storage)
    with patch("app.api.routes_projects.start_pipeline") as start:
        resp = client.post(f"/api/projects/{project_id}/process", json={"source_language": "zh"})
    assert resp.status_code == 422
    start.assert_not_called()


def test_process_rejects_voice_cloning_when_the_deployment_cannot_serve_it(client, fake_storage, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "tts_provider", "real")
    monkeypatch.setattr(get_settings(), "tts_voice_clone", False)
    project_id = _uploaded_project(client, fake_storage)
    with patch("app.api.routes_projects.start_pipeline") as start:
        resp = client.post(f"/api/projects/{project_id}/process", json={"clone_voice": True})
    assert resp.status_code == 422
    start.assert_not_called()
    assert client.get("/api/capabilities").json()["voice_clone_available"] is False


def test_language_gate_accepts_the_majority_detection_not_the_first_row():
    from types import SimpleNamespace

    from app.pipeline.tasks import detected_language_majority

    segs = [
        SimpleNamespace(detected_language="ta", source_text="short"),
        SimpleNamespace(detected_language="te", source_text="line one"),
        SimpleNamespace(detected_language="te", source_text="line two"),
        SimpleNamespace(detected_language="ja", source_text=""),
        SimpleNamespace(detected_language="ja", source_text=" "),
    ]
    assert detected_language_majority(segs) == "te"


# --- regenerate runs each half on the worker that has its models ------------
def test_regenerate_translates_on_the_translate_queue_and_renders_on_the_tts_queue():
    """regenerate used to be one task on q.synthesize that also translated;
    that worker runs the TTS venv, which has no IndicTrans2."""
    from app.celery_app import QUEUE_TRANSLATE, QUEUE_TTS
    from app.pipeline.regenerate import regenerate_segment, regenerate_synthesize

    assert regenerate_segment.queue == QUEUE_TRANSLATE
    assert regenerate_synthesize.queue == QUEUE_TTS


# --- infrastructure --------------------------------------------------------
def test_local_storage_download_of_a_missing_key_raises(tmp_path):
    """It returned silently, so the failure surfaced later as an unrelated
    ffmpeg "Invalid data found" error instead of naming the missing key."""
    from app.storage.local import LocalStorage

    storage = LocalStorage(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="never-written"):
        storage.download_file("projects/x/never-written.wav", str(tmp_path / "out.wav"))


def test_local_storage_rejects_keys_that_escape_the_root(tmp_path):
    from app.storage.local import LocalStorage

    with pytest.raises(ValueError, match="escapes"):
        LocalStorage(str(tmp_path / "root")).exists("../outside.txt")


def test_local_storage_root_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    from app.config import BACKEND_ROOT
    from app.storage.local import LocalStorage

    monkeypatch.chdir(tmp_path)
    assert LocalStorage().root == (BACKEND_ROOT / "data" / "storage").resolve()


def test_a_missing_ffmpeg_binary_is_a_permanent_error_naming_the_setting(monkeypatch):
    from app.config import get_settings
    from app.pipeline import ffmpeg_utils

    monkeypatch.setattr(get_settings(), "ffprobe_binary", "no/such/ffprobe.exe")
    with pytest.raises(ffmpeg_utils.MediaToolMissingError, match="FFPROBE_BINARY") as e:
        ffmpeg_utils.probe_duration_ms("whatever.mp4")
    assert isinstance(e.value, pipeline_tasks._PERMANENT)


def test_hf_offline_mode_needs_no_token(monkeypatch):
    from app.config import require_hf_token
    from app.startup_checks import verify_secrets

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert require_hf_token("TRANSLATION_PROVIDER=real") is None

    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "translation_provider", "real")
    verify_secrets()  # does not raise


def test_emotion_prosody_is_weighted_by_confidence_and_neutral_below_the_floor():
    from app.providers.base import EmotionResult
    from app.providers.tts.mms_provider import EMOTION_PROSODY, Prosody, prosody_for

    assert prosody_for(None, 0.4) == Prosody()
    assert prosody_for(EmotionResult("anger", 0.3), 0.4) == Prosody()
    full, anger = prosody_for(EmotionResult("anger", 1.0), 0.4), EMOTION_PROSODY["anger"]
    assert (full.rate, full.gain_db, full.noise_delta) == pytest.approx(
        (anger.rate, anger.gain_db, anger.noise_delta))
    half = prosody_for(EmotionResult("sadness", 0.7), 0.4)
    assert half.rate == pytest.approx(1 + (EMOTION_PROSODY["sadness"].rate - 1) * 0.5)
    with pytest.raises(ValueError):
        prosody_for(EmotionResult("contempt", 0.9), 0.4)


def test_every_tts_capable_language_names_a_voice():
    from app.capabilities import SUPPORTED_LANGUAGES

    for lang in SUPPORTED_LANGUAGES:
        assert lang.tts_supported == bool(lang.mms_tts)
