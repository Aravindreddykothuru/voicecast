"""One Celery task per pipeline stage (PRD section 07).

Every task:
  * is idempotent and resumable -- it re-derives the work left from Postgres
    rather than assuming a clean slate, and the per-segment stages commit
    each segment as it finishes. That second part matters on CPU: synthesize
    used to hold every segment in one transaction, so a failure on segment
    17 of 20 rolled back 16 finished renders (minutes each) and the retry
    started from zero. Now a retry picks up at segment 17, and the UI sees
    segments land as they are rendered instead of all at the very end;
  * only ever touches models/providers/storage -- never a model library
    directly (see app/providers/base.py's docstring for why);
  * emits stage_started/stage_progress/stage_completed around its work and
    error on failure, over the same EventBus the WebSocket gateway reads;
  * binds project_id (and segment_id where relevant) into the logging
    context so every log line for a run is greppable by project_id.

Segments with no speech (ASR returned no text -- a laugh, music, a VAD false
positive) flow through every stage as explicit no-ops: nothing is translated
or spoken for them and the mux leaves their stretch silent. Translating an
empty string and asking TTS to say it produced either a crash or a
hallucinated sentence.

`app/pipeline/chain.py` wires these into the Celery chain that /process
kicks off. `app/pipeline/regenerate.py` calls the per-segment helpers here
directly for the single-segment /regenerate endpoint, without re-running
extract/diarize/transcribe.
"""
from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone

from sqlalchemy import func, select

from app.celery_app import celery_app
from app.config import get_settings
from app.db import session_scope
from app.logging_conf import bind_context, clear_context
from app.models.base import is_uuid
from app.models.export_job import ExportJob, ExportStatus
from app.models.project import Project, ProjectStatus
from app.models.segment import EmotionLabel, Segment, SegmentStatus
from app.models.source_video import SourceVideo, SourceVideoStatus
from app.models.speaker import Speaker
from app.pipeline import ffmpeg_utils
from app.pipeline.events import (
    emit_error,
    emit_segment_ready,
    emit_stage_completed,
    emit_stage_progress,
    emit_stage_started,
)
from app.pipeline import liveness
from app.pipeline.text_normalize import spell_out_numbers_en
from app.pipeline.timeline import GUARD_MS, TimelineSlot, normalize_turns, plan_timeline
from app.providers.base import EmotionResult, SpeakerChunk, SynthesisRequest
from app.providers.registry import (
    get_asr_provider,
    get_diarization_provider,
    get_emotion_provider,
    get_translation_provider,
    get_tts_provider,
)
from app.storage import get_storage

logger = logging.getLogger(__name__)

# Every stage task heartbeats while it runs (see liveness.py).
liveness.connect_signals()

_RETRYABLE = (Exception,)

# Permanent failures: a missing project/segment/audio row, or an unsupported
# target language, will fail exactly the same way on every attempt, so
# retrying them three times with backoff only delays the error the operator
# needs to see and holds a worker slot while doing it. Everything else
# (network blips, S3 timeouts, a flaky ffmpeg subprocess) stays retryable.
_PERMANENT = (ValueError, LookupError, TypeError)


def _target_lang(project: Project) -> str:
    return project.target_languages[0] if project.target_languages else get_settings().default_target_language


def has_speech(segment: Segment) -> bool:
    return bool((segment.source_text or "").strip())


def _mark_project_failed(project_id: str, stage: str, exc: Exception) -> None:
    with session_scope() as db:
        # Guard first: this runs inside the exception handler, so a DataError
        # from a malformed id would replace the real failure with a confusing
        # one and lose the original error entirely.
        project = db.get(Project, project_id) if is_uuid(project_id) else None
        if project:
            project.status = ProjectStatus.failed
            project.error_message = f"{stage}: {exc}"[:2000]
            project.error_is_permanent = isinstance(exc, _PERMANENT)
    emit_error(project_id, stage, str(exc), permanent=isinstance(exc, _PERMANENT))


def _set_stage(db, project: Project, stage: str) -> None:
    project.current_stage = stage
    if project.status not in (ProjectStatus.processing,):
        project.status = ProjectStatus.processing
    # A stage only starts once the previous one succeeded, so any stored
    # error is necessarily resolved by now -- leaving it would let a stale,
    # already-fixed failure keep showing up in anything that reads
    # error_message without also checking status == failed.
    project.error_message = None
    project.error_is_permanent = None


def _require_project(db, project_id: str) -> Project:
    """Fail loudly and legibly if the project vanished mid-pipeline, rather
    than letting every stage trip over `NoneType has no attribute ...`."""
    # ValueError, not DataError: a malformed id can never become valid, and
    # ValueError is in _PERMANENT so Celery won't burn retries on it.
    project = db.get(Project, project_id) if is_uuid(project_id) else None
    if project is None:
        raise ValueError(f"project {project_id} not found")
    return project


def _require_audio_key(db, source_video_id: str) -> str:
    """Returns the extracted-audio storage key, or fails with a clear message."""
    video = db.get(SourceVideo, source_video_id) if is_uuid(source_video_id) else None
    if video is None:
        raise ValueError(f"source video {source_video_id} not found")
    if not video.audio_storage_key:
        raise ValueError(f"source video {source_video_id} has no extracted audio")
    return video.audio_storage_key


def _segment_ids(db, project_id: str, statuses: Iterable[SegmentStatus]) -> list[str]:
    return list(
        db.execute(
            select(Segment.id)
            .where(Segment.project_id == project_id, Segment.status.in_(list(statuses)))
            .order_by(Segment.index)
        ).scalars()
    )


@contextlib.contextmanager
def _downloaded(storage, key: str, name: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name)
        storage.download_file(key, path)
        yield path


# ---------------------------------------------------------------------------
# Stage 1: extract_audio
# ---------------------------------------------------------------------------
@celery_app.task(bind=True, name="app.pipeline.tasks.extract_audio", autoretry_for=_RETRYABLE, dont_autoretry_for=_PERMANENT, max_retries=3,
                  retry_backoff=True)
def extract_audio(self, project_id: str, source_video_id: str) -> str:
    bind_context(project_id=project_id)
    stage = "extract_audio"
    emit_stage_started(project_id, stage)
    try:
        with session_scope() as db:
            project = db.get(Project, project_id) if is_uuid(project_id) else None
            video = db.get(SourceVideo, source_video_id) if is_uuid(source_video_id) else None
            if project is None or video is None:
                raise ValueError("project or source video not found")
            _set_stage(db, project, stage)

            if video.audio_storage_key and video.status == SourceVideoStatus.extracted:
                logger.info("extract_audio: already extracted, skipping (idempotent)")
                emit_stage_completed(project_id, stage)
                return project_id

            storage = get_storage()
            with tempfile.TemporaryDirectory() as tmp:
                video_path = os.path.join(tmp, "source.mp4")
                audio_path = os.path.join(tmp, "audio.wav")
                storage.download_file(video.storage_key, video_path)
                ffmpeg_utils.extract_audio_wav(video_path, audio_path)

                duration_ms = ffmpeg_utils.probe_duration_ms(video_path)
                audio_key = f"projects/{project_id}/videos/{source_video_id}/audio.wav"
                storage.upload_file(audio_key, audio_path, content_type="audio/wav")

            video.audio_storage_key = audio_key
            video.duration_ms = duration_ms or video.duration_ms
            video.status = SourceVideoStatus.extracted

        emit_stage_progress(project_id, stage, 1.0)
        emit_stage_completed(project_id, stage)
        return project_id
    except Exception as exc:  # noqa: BLE001
        _mark_project_failed(project_id, stage, exc)
        raise
    finally:
        clear_context()


# ---------------------------------------------------------------------------
# Stage 2: chunk_and_diarize
# ---------------------------------------------------------------------------
# No segment may exceed this. A voice-cloned render uses the segment's own
# audio slice as the CosyVoice2 reference, and CosyVoice2 hard-asserts that
# reference is <= 30s ("do not support extract speech token for audio longer
# than 30s"). Real diarization happily returns a 40s monologue turn, so the
# cap is applied to every chunk -- provider output and the no-speech fallback
# alike -- not just one of them. 25s leaves headroom under the 30s limit.
MAX_SEGMENT_MS = 25_000


def _cap_chunk_lengths(chunks: list[SpeakerChunk], max_ms: int) -> list[SpeakerChunk]:
    """Split any chunk longer than `max_ms` into contiguous sub-chunks,
    preserving speaker_tag and leaving shorter chunks untouched."""
    out: list[SpeakerChunk] = []
    for c in chunks:
        span = c.end_ms - c.start_ms
        if span <= max_ms:
            out.append(c)
            continue
        for start in range(c.start_ms, c.end_ms, max_ms):
            out.append(
                SpeakerChunk(
                    start_ms=start,
                    end_ms=min(start + max_ms, c.end_ms),
                    speaker_tag=c.speaker_tag,
                )
            )
    return out


@celery_app.task(bind=True, name="app.pipeline.tasks.chunk_and_diarize", autoretry_for=_RETRYABLE, dont_autoretry_for=_PERMANENT, max_retries=3,
                  retry_backoff=True)
def chunk_and_diarize(self, project_id: str, source_video_id: str) -> str:
    bind_context(project_id=project_id)
    stage = "chunk_and_diarize"
    emit_stage_started(project_id, stage)
    try:
        with session_scope() as db:
            project = db.get(Project, project_id) if is_uuid(project_id) else None
            video = db.get(SourceVideo, source_video_id) if is_uuid(source_video_id) else None
            if project is None or video is None or not video.audio_storage_key:
                raise ValueError("source video audio not extracted yet")
            _set_stage(db, project, stage)

            existing = db.execute(
                select(Segment.id).where(Segment.source_video_id == source_video_id).limit(1)
            ).first()
            if existing:
                logger.info("chunk_and_diarize: segments already exist, skipping (idempotent)")
                emit_stage_completed(project_id, stage)
                return project_id

            storage = get_storage()
            with _downloaded(storage, video.audio_storage_key, "audio.wav") as audio_path:
                raw = get_diarization_provider().chunk_and_diarize(audio_path)

            # Overlapping turns become overlapping dubbed voices; same-speaker
            # fragments become ASR calls on half a sentence. See timeline.py.
            chunks = normalize_turns(raw)
            if len(chunks) != len(raw):
                logger.info("chunk_and_diarize: normalized %d raw turns into %d", len(raw), len(chunks))

            if not chunks:
                # Diarization found no speech turns. On a genuinely empty or
                # music-only file this is correct; on a short/quiet
                # single-speaker clip pyannote's VAD just misses. Either way,
                # producing zero segments silently strands the run in
                # "processing" forever (transcribe early-returns on no
                # segments). Fall back to one whole-file chunk (the length cap
                # below windows it); raise only when we can't even bound it.
                # See CONTRACTS.md #5 (no silent dead-ends).
                duration_ms = video.duration_ms or 0
                if duration_ms <= 0:
                    raise ValueError(
                        "diarization produced no speech segments and the audio "
                        "duration is unknown -- the source file may be silent, "
                        "music-only, or corrupt."
                    )
                logger.warning(
                    "chunk_and_diarize: no diarization turns; falling back to a "
                    "whole-file segment (0-%dms), windowed below.",
                    duration_ms,
                )
                chunks = [SpeakerChunk(start_ms=0, end_ms=duration_ms, speaker_tag="SPEAKER_00")]

            before = len(chunks)
            chunks = _cap_chunk_lengths(chunks, MAX_SEGMENT_MS)
            if len(chunks) != before:
                logger.info(
                    "chunk_and_diarize: split %d long turn(s) into %d segments "
                    "(cap %dms) so each stays a valid TTS voice reference.",
                    before,
                    len(chunks),
                    MAX_SEGMENT_MS,
                )

            speakers_by_tag: dict[str, Speaker] = {}
            for i, chunk in enumerate(chunks):
                speaker = speakers_by_tag.get(chunk.speaker_tag)
                if speaker is None:
                    speaker = db.execute(
                        select(Speaker).where(
                            Speaker.project_id == project_id, Speaker.diarization_tag == chunk.speaker_tag
                        )
                    ).scalar_one_or_none()
                    if speaker is None:
                        speaker = Speaker(
                            project_id=project_id,
                            label=f"Speaker {len(speakers_by_tag) + 1}",
                            diarization_tag=chunk.speaker_tag,
                        )
                        db.add(speaker)
                        db.flush()
                    speakers_by_tag[chunk.speaker_tag] = speaker

                db.add(
                    Segment(
                        project_id=project_id,
                        source_video_id=source_video_id,
                        speaker_id=speaker.id,
                        index=i,
                        start_ms=chunk.start_ms,
                        end_ms=chunk.end_ms,
                        status=SegmentStatus.pending,
                    )
                )
                emit_stage_progress(project_id, stage, (i + 1) / max(len(chunks), 1), completed=i + 1, total=len(chunks))

        emit_stage_completed(project_id, stage)
        return project_id
    except Exception as exc:  # noqa: BLE001
        _mark_project_failed(project_id, stage, exc)
        raise
    finally:
        clear_context()


# ---------------------------------------------------------------------------
# Stage 3: transcribe
# ---------------------------------------------------------------------------
def _transcribe_one(audio_path: str, segment: Segment, language: str | None = None) -> None:
    bind_context(segment_id=segment.id)
    result = get_asr_provider().transcribe_chunk(
        audio_path, segment.start_ms, segment.end_ms, language=language
    )
    segment.source_text = result.text
    segment.detected_language = result.language
    segment.detected_language_confidence = result.confidence
    segment.status = SegmentStatus.transcribed


@celery_app.task(bind=True, name="app.pipeline.tasks.transcribe", autoretry_for=_RETRYABLE, dont_autoretry_for=_PERMANENT, max_retries=3,
                  retry_backoff=True)
def transcribe(self, project_id: str, retranscribe: bool = False) -> str:
    """Transcribe pending segments; with `retranscribe`, also redo finished ones.

    It used to always re-pick already-transcribed segments AND raise when it
    found nothing to do. Restarting a run that failed during synthesis --
    the dashboard's Retry -- therefore died right here ("no segments to
    transcribe"), because every segment was already past this stage.
    """
    bind_context(project_id=project_id)
    stage = "transcribe"
    emit_stage_started(project_id, stage)
    try:
        with session_scope() as db:
            project = _require_project(db, project_id)
            _set_stage(db, project, stage)
            language = project.source_language
            statuses = (SegmentStatus.pending, SegmentStatus.transcribed) if retranscribe else (SegmentStatus.pending,)
            ids = _segment_ids(db, project_id, statuses)
            total = db.execute(
                select(func.count(Segment.id)).where(Segment.project_id == project_id)
            ).scalar_one()
            if total == 0:
                # Not a benign "nothing to do": every earlier stage completed
                # yet no segment exists, so the run would otherwise strand in
                # "processing" with no error. Fail loudly instead.
                raise ValueError(
                    "transcribe: no segments to transcribe -- diarization "
                    "produced nothing for this project."
                )
            audio_key = _require_audio_key(db, db.get(Segment, ids[0]).source_video_id) if ids else None
            if not ids:
                logger.info("transcribe: all %d segments already transcribed; resuming", total)

        if ids:
            with _downloaded(get_storage(), audio_key, "audio.wav") as audio_path:
                for i, seg_id in enumerate(ids):
                    with session_scope() as db:
                        _transcribe_one(audio_path, db.get(Segment, seg_id), language)
                    emit_stage_progress(project_id, stage, (i + 1) / len(ids), completed=i + 1, total=len(ids))

        with session_scope() as db:
            project = _require_project(db, project_id)
            if project.review_language and not project.source_language:
                # Park here. The expensive half only starts once someone has
                # confirmed the detected language via /confirm-language.
                # (A confirmed language means it already passed the gate --
                # a restarted run must not ask again.)
                project.status = ProjectStatus.awaiting_language_confirmation
                project.current_stage = None
                logger.info("transcribe: awaiting source-language confirmation")

        emit_stage_completed(project_id, stage)
        return project_id
    except Exception as exc:  # noqa: BLE001
        _mark_project_failed(project_id, stage, exc)
        raise
    finally:
        clear_context()


# ---------------------------------------------------------------------------
# Stage 4: detect_emotion (P2 -- skipped, defaulted to neutral, if the
# project's preserve_emotion flag is off)
# ---------------------------------------------------------------------------
def _detect_emotion_one(audio_path: str, segment: Segment) -> None:
    bind_context(segment_id=segment.id)
    result = get_emotion_provider().detect(audio_path, segment.start_ms, segment.end_ms)
    segment.emotion_label = EmotionLabel(result.label)
    segment.emotion_score = result.score
    segment.status = SegmentStatus.emotion_detected


@celery_app.task(bind=True, name="app.pipeline.tasks.detect_emotion", autoretry_for=_RETRYABLE, dont_autoretry_for=_PERMANENT, max_retries=3,
                  retry_backoff=True)
def detect_emotion(self, project_id: str) -> str:
    bind_context(project_id=project_id)
    stage = "detect_emotion"
    emit_stage_started(project_id, stage)
    try:
        with session_scope() as db:
            project = _require_project(db, project_id)
            _set_stage(db, project, stage)
            preserve = project.preserve_emotion
            ids = _segment_ids(db, project_id, (SegmentStatus.transcribed,))
            if not ids:
                emit_stage_completed(project_id, stage)
                return project_id
            audio_key = _require_audio_key(db, db.get(Segment, ids[0]).source_video_id) if preserve else None

        with contextlib.ExitStack() as stack:
            audio_path = stack.enter_context(_downloaded(get_storage(), audio_key, "audio.wav")) if audio_key else None
            for i, seg_id in enumerate(ids):
                with session_scope() as db:
                    seg = db.get(Segment, seg_id)
                    if not has_speech(seg):
                        # Nothing was said, so there is no register to carry.
                        seg.emotion_label = None
                        seg.emotion_score = None
                        seg.status = SegmentStatus.emotion_detected
                    elif not preserve:
                        seg.emotion_label = EmotionLabel.neutral
                        seg.emotion_score = 1.0
                        seg.status = SegmentStatus.emotion_detected
                    else:
                        _detect_emotion_one(audio_path, seg)
                emit_stage_progress(project_id, stage, (i + 1) / len(ids), completed=i + 1, total=len(ids))
        emit_stage_completed(project_id, stage)
        return project_id
    except Exception as exc:  # noqa: BLE001
        _mark_project_failed(project_id, stage, exc)
        raise
    finally:
        clear_context()


# ---------------------------------------------------------------------------
# Stage 5: translate
# ---------------------------------------------------------------------------
def translate_segment(segment: Segment, target_lang: str, source_lang: str) -> None:
    """Shared by the full-project stage task and single-segment /regenerate.

    `source_lang` MUST be passed explicitly -- TranslationProvider.translate's
    src_lang default of "en" existed for call-site convenience, and every real
    call site silently relied on it instead of ever passing what ASR actually
    detected. That is the bug: a segment detected as Chinese at 98% confidence
    got translated as if it were English, with no error anywhere. See
    CONTRACTS.md #3 and #5.
    """
    bind_context(segment_id=segment.id)
    if not has_speech(segment):
        segment.translated_text = ""
    else:
        text = segment.source_text or ""
        if source_lang == "en":
            # Numerals would otherwise reach the TTS voice, which can't say
            # them. See app/pipeline/text_normalize.py.
            text = spell_out_numbers_en(text)
        result = get_translation_provider().translate(text, target_lang, src_lang=source_lang)
        segment.translated_text = result.text
    segment.status = SegmentStatus.translated


def detected_language_majority(segments: Iterable) -> str | None:
    """Most common ASR-detected language across segments that contain speech.

    Segments with no text are excluded: Whisper's language guess on silence
    or music is noise, and on a clip with a few such gaps it could outvote
    the real language. Shared by the translate stage and the language gate
    (/confirm-language), which used to take whichever segment row came back
    first instead.
    """
    detected = [
        s.detected_language
        for s in segments
        if s.detected_language and (getattr(s, "source_text", None) is None or has_speech(s))
    ]
    return Counter(detected).most_common(1)[0][0] if detected else None


def _project_source_language(project: Project, segments: list[Segment]) -> str:
    """The confirmed source language for this project, or the ASR majority
    vote if the confirmation gate was skipped (review_language=False).

    Never falls back to a hardcoded "en": that silent substitution is exactly
    what let a 98%-confidence Chinese detection be translated as English.
    Raises if no source language is known at all, which the caller lets
    surface as a permanent (non-retried) failure. See CONTRACTS.md #3 and #5.
    """
    if project.source_language:
        return project.source_language
    majority = detected_language_majority(segments)
    if majority:
        return majority
    raise ValueError(
        f"project {project.id} has no confirmed source_language and no segment "
        "reported a detected_language -- cannot translate without guessing."
    )


@celery_app.task(bind=True, name="app.pipeline.tasks.translate", autoretry_for=_RETRYABLE, dont_autoretry_for=_PERMANENT, max_retries=3,
                  retry_backoff=True)
def translate(self, project_id: str) -> str:
    bind_context(project_id=project_id)
    stage = "translate"
    emit_stage_started(project_id, stage)
    try:
        with session_scope() as db:
            project = _require_project(db, project_id)
            _set_stage(db, project, stage)
            target_lang = _target_lang(project)
            all_segments = db.execute(
                select(Segment).where(Segment.project_id == project_id).order_by(Segment.index)
            ).scalars().all()
            if all_segments and not any(has_speech(s) for s in all_segments):
                raise ValueError(
                    "no speech was transcribed in any segment -- there is nothing to dub. "
                    "The source may be silent or music-only."
                )

            # Resolved and validated ONCE, before spending any translation
            # calls: an unsupported source or target language fails the whole
            # stage immediately and clearly instead of burning partial work.
            source_lang = _project_source_language(project, all_segments)
            from app.capabilities import require_language, require_source_language

            require_source_language(source_lang)
            require_language(target_lang)
            ids = _segment_ids(db, project_id, (SegmentStatus.emotion_detected,))

        for i, seg_id in enumerate(ids):
            with session_scope() as db:
                translate_segment(db.get(Segment, seg_id), target_lang, source_lang)
            emit_stage_progress(project_id, stage, (i + 1) / max(len(ids), 1), completed=i + 1, total=len(ids))
        emit_stage_completed(project_id, stage)
        return project_id
    except Exception as exc:  # noqa: BLE001
        _mark_project_failed(project_id, stage, exc)
        raise
    finally:
        clear_context()


# ---------------------------------------------------------------------------
# Stage 6: synthesize
# ---------------------------------------------------------------------------
# How much of a speaker's own speech to give voice conversion as its
# reference. CosyVoice2 caps the reference at 30s.
REFERENCE_TARGET_MS = 12_000
REFERENCE_MAX_MS = 28_000


def _voice_reference(db, storage, segment: Segment, tmp: str) -> str:
    """A local WAV of this segment's speaker, for voice cloning.

    One reference per speaker, built once from that speaker's longest spoken
    segments (~12s) and stored on the Speaker row, so every line a speaker
    says is converted toward the same voice. It used to be each segment's
    own audio slice: a one-word line got a 0.5s reference, every line got a
    different one, and on the real run the cloned dub read back at corpus
    CER 0.30 against 0.16 for the same dub unconverted.

    Raises rather than quietly rendering in a stock voice: clone_voice was
    explicitly requested.
    """
    path = os.path.join(tmp, "reference_clip.wav")
    speaker = db.get(Speaker, segment.speaker_id) if segment.speaker_id else None
    if speaker and speaker.reference_clip_url:
        storage.download_file(speaker.reference_clip_url, path)
        return path

    candidates = [segment]
    if speaker is not None:
        candidates = db.execute(
            select(Segment).where(Segment.project_id == segment.project_id, Segment.speaker_id == speaker.id)
        ).scalars().all()
    spoken = [s for s in candidates if has_speech(s)] or [segment]
    chosen, total = [], 0
    for s in sorted(spoken, key=lambda s: s.end_ms - s.start_ms, reverse=True):
        span = s.end_ms - s.start_ms
        if total >= REFERENCE_TARGET_MS:
            break
        if total + span > REFERENCE_MAX_MS:
            continue
        chosen.append(s)
        total += span
    chosen.sort(key=lambda s: s.start_ms)

    full_audio = os.path.join(tmp, "full_audio.wav")
    storage.download_file(_require_audio_key(db, segment.source_video_id), full_audio)
    pieces = []
    for i, s in enumerate(chosen):
        with ffmpeg_utils.extract_audio_slice(full_audio, s.start_ms, s.end_ms) as slice_path:
            piece = os.path.join(tmp, f"ref_{i:02d}.wav")
            shutil.copyfile(slice_path, piece)
            pieces.append(piece)
    ffmpeg_utils.concat_audio(pieces, path)

    if speaker is not None:
        key = f"projects/{segment.project_id}/speakers/{speaker.id}/reference.wav"
        storage.upload_file(key, path, content_type="audio/wav")
        speaker.reference_clip_url = key
    return path


def _available_ms(db, segment: Segment, original_ms: int) -> int:
    later = db.execute(
        select(Segment.start_ms, Segment.source_text)
        .where(Segment.project_id == segment.project_id, Segment.start_ms > segment.start_ms)
        .order_by(Segment.start_ms)
    ).all()
    next_start = next((start for start, text in later if (text or "").strip()), None)
    if next_start is None:
        video = db.get(SourceVideo, segment.source_video_id)
        next_start = video.duration_ms if video and video.duration_ms else segment.end_ms
    return max(next_start - segment.start_ms - GUARD_MS, original_ms)


def _speaker_voice_gender(db, storage, segment: Segment, tmp: str) -> str | None:
    """The speaker's voice gender, estimated once from their longest line and
    stored, when the TTS engine has voices to choose between. Without it
    every speaker in a scene was dubbed by the same single voice."""
    settings = get_settings()
    if settings.tts_provider != "real" or settings.tts_engine != "syspin" or not segment.speaker_id:
        return None
    speaker = db.get(Speaker, segment.speaker_id)
    if speaker is None:
        return None
    if speaker.voice_gender is None:
        spoken = db.execute(
            select(Segment).where(Segment.project_id == segment.project_id, Segment.speaker_id == speaker.id)
        ).scalars().all()
        longest = max((s for s in spoken if has_speech(s)), key=lambda s: s.end_ms - s.start_ms, default=segment)
        import librosa

        from app.providers.tts.common import estimate_voice_gender

        full_audio = os.path.join(tmp, "gender_source.wav")
        storage.download_file(_require_audio_key(db, segment.source_video_id), full_audio)
        with ffmpeg_utils.extract_audio_slice(full_audio, longest.start_ms, min(longest.end_ms, longest.start_ms + 10_000)) as p:
            audio, sr = librosa.load(p, sr=16000)
        speaker.voice_gender = estimate_voice_gender(audio, sr) or "unknown"
        logger.info("speaker %s voice gender: %s", speaker.label, speaker.voice_gender)
    return speaker.voice_gender if speaker.voice_gender != "unknown" else None


def synthesize_segment(db, storage, segment: Segment, project: Project) -> None:
    """Shared by the full-project stage task and single-segment /regenerate."""
    bind_context(segment_id=segment.id)
    text = (segment.translated_text or "").strip()
    if not text:
        # No speech in, no speech out: the mux leaves this stretch silent.
        segment.tts_audio_url = None
        segment.tts_duration_ms = None
        segment.sync_offset_pct = None
        segment.status = SegmentStatus.synthesized
        return

    with tempfile.TemporaryDirectory() as tmp:
        voice_ref_path = voice_ref_text = None
        if project.clone_voice:
            voice_ref_path = _voice_reference(db, storage, segment, tmp)
            voice_ref_text = segment.source_text

        emotion = None
        if project.preserve_emotion and segment.emotion_label:
            emotion = EmotionResult(label=segment.emotion_label.value, score=segment.emotion_score or 0.0)

        extra = {"tts_model": project.tts_model} if project.tts_model else {}
        gender = _speaker_voice_gender(db, storage, segment, tmp)
        if gender:
            extra["voice_gender"] = gender

        original_ms = max(segment.end_ms - segment.start_ms, 1)
        request = SynthesisRequest(
            text=text,
            target_lang=_target_lang(project),
            voice_reference_path=voice_ref_path,
            voice_reference_text=voice_ref_text,
            emotion=emotion,
            # The time this line may occupy: up to the next spoken line, the
            # same window the mux planner allows (app/pipeline/timeline.py).
            # It used to be the segment's own length, so TTS squeezed a line
            # to fit even when a pause followed that the mux would happily
            # have used -- measurably less intelligible speech for nothing.
            target_duration_ms=_available_ms(db, segment, original_ms),
            extra=extra,
        )
        result = get_tts_provider().synthesize(request)
        try:
            key = f"projects/{project.id}/segments/{segment.id}/tts.wav"
            storage.upload_file(key, result.local_audio_path, content_type="audio/wav")
        finally:
            with contextlib.suppress(OSError):
                os.remove(result.local_audio_path)

    segment.tts_audio_url = key
    segment.tts_duration_ms = result.duration_ms
    segment.sync_offset_pct = round(100.0 * (result.duration_ms - original_ms) / original_ms, 2)
    segment.status = SegmentStatus.synthesized


@celery_app.task(bind=True, name="app.pipeline.tasks.synthesize", autoretry_for=_RETRYABLE, dont_autoretry_for=_PERMANENT, max_retries=3,
                  retry_backoff=True)
def synthesize(self, project_id: str) -> str:
    bind_context(project_id=project_id)
    stage = "synthesize"
    emit_stage_started(project_id, stage)
    try:
        with session_scope() as db:
            project = _require_project(db, project_id)
            _set_stage(db, project, stage)
            ids = _segment_ids(db, project_id, (SegmentStatus.translated,))

        storage = get_storage()
        for i, seg_id in enumerate(ids):
            with session_scope() as db:
                seg = db.get(Segment, seg_id)
                synthesize_segment(db, storage, seg, db.get(Project, project_id))
            emit_segment_ready(project_id, seg_id)
            emit_stage_progress(project_id, stage, (i + 1) / max(len(ids), 1), completed=i + 1, total=len(ids))
        emit_stage_completed(project_id, stage)
        return project_id
    except Exception as exc:  # noqa: BLE001
        _mark_project_failed(project_id, stage, exc)
        raise
    finally:
        clear_context()


# ---------------------------------------------------------------------------
# Stage 7: mux_export
# ---------------------------------------------------------------------------
@celery_app.task(bind=True, name="app.pipeline.tasks.mux_export", autoretry_for=_RETRYABLE, dont_autoretry_for=_PERMANENT, max_retries=3,
                  retry_backoff=True)
def mux_export(self, project_id: str) -> str:
    bind_context(project_id=project_id)
    stage = "mux_export"
    emit_stage_started(project_id, stage)
    try:
        with session_scope() as db:
            project = _require_project(db, project_id)
            _set_stage(db, project, stage)
            segments = db.execute(
                select(Segment)
                .where(Segment.project_id == project_id)
                .order_by(Segment.index)
            ).scalars().all()
            if not segments or any(s.status not in (SegmentStatus.synthesized, SegmentStatus.muxed) for s in segments):
                raise ValueError("not all segments are synthesized yet")
            if not any(s.tts_audio_url for s in segments):
                raise ValueError("no segment produced any speech to mux")

            source_video = db.get(SourceVideo, segments[0].source_video_id)
            if source_video is None:
                raise ValueError("source video not found for these segments")
            storage = get_storage()

            export = db.execute(
                select(ExportJob).where(
                    ExportJob.project_id == project_id, ExportJob.status != ExportStatus.ready
                )
            ).scalars().first()
            if export is None:
                export = ExportJob(project_id=project_id, status=ExportStatus.running)
                db.add(export)
                db.flush()
            export.status = ExportStatus.running

            with tempfile.TemporaryDirectory() as tmp:
                video_path = os.path.join(tmp, "source.mp4")
                storage.download_file(source_video.storage_key, video_path)
                duration_ms = source_video.duration_ms or ffmpeg_utils.probe_duration_ms(video_path)

                plan = plan_timeline(
                    [
                        TimelineSlot(s.id, s.start_ms, s.end_ms, s.tts_duration_ms if s.tts_audio_url else None)
                        for s in segments
                    ],
                    duration_ms,
                )
                by_id = {s.id: s for s in segments}
                placements: list[ffmpeg_utils.ClipPlacement] = []
                for i, clip in enumerate(plan):
                    path = os.path.join(tmp, f"seg_{i:04d}.wav")
                    storage.download_file(by_id[clip.segment_id].tts_audio_url, path)
                    placements.append(ffmpeg_utils.ClipPlacement(path, clip.start_ms, clip.tempo))
                    emit_stage_progress(project_id, stage, 0.1 + 0.6 * (i + 1) / len(plan))
                    if clip.overrun_ms:
                        logger.warning(
                            "mux_export: segment %s still runs %dms past its window at tempo %.2f",
                            clip.segment_id, clip.overrun_ms, clip.tempo,
                        )

                out_path = os.path.join(tmp, "output.mp4")
                ffmpeg_utils.mux_timeline(video_path, placements, out_path)

                out_key = f"projects/{project_id}/exports/{export.id}/output.mp4"
                storage.upload_file(out_key, out_path, content_type="video/mp4")

            export.output_url = out_key
            export.qa_report = _qa_report(segments, plan, duration_ms)
            export.status = ExportStatus.ready
            export.completed_at = datetime.now(timezone.utc)

            for s in segments:
                s.status = SegmentStatus.muxed
            project.status = ProjectStatus.ready
            project.current_stage = None
            # Clear any failure recorded by an earlier attempt -- otherwise a
            # project that later succeeds still reports the stale error to the
            # dashboard alongside status="ready".
            project.error_message = None
            project.error_is_permanent = None

        emit_stage_progress(project_id, stage, 1.0)
        emit_stage_completed(project_id, stage)
        return project_id
    except Exception as exc:  # noqa: BLE001
        _mark_project_failed(project_id, stage, exc)
        raise
    finally:
        clear_context()


def _qa_report(segments: list[Segment], plan, duration_ms: int) -> dict:
    fits = {c.segment_id: c for c in plan}
    voiced = [s for s in segments if s.tts_audio_url]
    rows = []
    for s in segments:
        fit = fits.get(s.id)
        rows.append(
            {
                "segment_id": s.id,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "has_speech": fit is not None,
                "sync_offset_pct": s.sync_offset_pct,
                "emotion_label": s.emotion_label.value if s.emotion_label else None,
                "tempo": fit.tempo if fit else None,
                "fitted_ms": fit.fitted_ms if fit else None,
                "overrun_ms": fit.overrun_ms if fit else None,
            }
        )
    offsets = [abs(s.sync_offset_pct) for s in voiced if s.sync_offset_pct is not None]
    return {
        "segments": rows,
        "overall": {
            "segment_count": len(segments),
            "speech_segment_count": len(voiced),
            "duration_ms": duration_ms,
            "avg_sync_offset_pct": round(sum(offsets) / len(offsets), 2) if offsets else 0.0,
            "max_tempo": max((c.tempo for c in plan), default=1.0),
            "overrun_segment_count": sum(1 for c in plan if c.overrun_ms),
            "total_overrun_ms": sum(c.overrun_ms for c in plan),
        },
    }
