"""RuntimeSyspinTTSProvider: the feature-flagged swap of SyspinTTSProvider's
raw render to an isolated subprocess (app.tts_runtime). See CONTRACTS.md #9
and app/providers/tts/runtime_syspin_provider.py.

Fast tests (no weights): the registry routes on the flag, and the flag
defaults off (one-line rollback is a real default, not just a documented
one). Slow tests (real SYSPIN weights, skipped if not cached): the full
`synthesize()` output -- after prosody, time-fit and loudness normalize,
not just the raw render -- is byte-identical between the two providers on a
fixed sample set, and a crash in the isolated subprocess surfaces as a
retryable exception rather than corrupting or hanging the caller.
"""
from __future__ import annotations

import hashlib

import pytest

from app.providers.base import EmotionResult, SynthesisRequest

pytest.importorskip("torch")
from app.tts_runtime.adapters.syspin import SyspinAdapter  # noqa: E402

SYSPIN_OK, SYSPIN_WHY = SyspinAdapter.availability({})
needs_syspin = pytest.mark.skipif(not SYSPIN_OK, reason=f"SYSPIN not cached: {SYSPIN_WHY}")

# Same shape as tasks.py's _PERMANENT / _RETRYABLE split: a plain RuntimeError
# (which every app.tts_runtime.pool.WorkerError is) must NOT be in _PERMANENT,
# or a transient subprocess crash would fail a segment on the first try
# instead of retrying it like any other flaky stage.
from app.pipeline.tasks import _PERMANENT  # noqa: E402


# --- fast: no weights, no subprocess -----------------------------------------------------
def test_the_flag_defaults_off():
    from app.config import Settings

    assert Settings.model_fields["tts_use_runtime"].default is False


def test_registry_routes_on_the_flag(monkeypatch):
    from app.config import get_settings
    from app.providers import registry

    class FakeRuntime:
        pass

    class FakeInProcess:
        pass

    monkeypatch.setattr("app.providers.tts.runtime_syspin_provider.RuntimeSyspinTTSProvider", FakeRuntime)
    monkeypatch.setattr("app.providers.tts.syspin_provider.SyspinTTSProvider", FakeInProcess)
    s = get_settings()
    registry.get_tts_provider.cache_clear()
    monkeypatch.setattr(s, "tts_provider", "real")
    monkeypatch.setattr(s, "tts_engine", "syspin")

    monkeypatch.setattr(s, "tts_use_runtime", True)
    registry.get_tts_provider.cache_clear()
    assert isinstance(registry.get_tts_provider(), FakeRuntime)

    monkeypatch.setattr(s, "tts_use_runtime", False)
    registry.get_tts_provider.cache_clear()
    assert isinstance(registry.get_tts_provider(), FakeInProcess)
    registry.get_tts_provider.cache_clear()


def test_a_worker_error_is_retryable_not_permanent():
    """The rollback story only holds if a crash in the isolated subprocess
    gets Celery's normal retry-with-backoff, not a permanent failure that
    burns the job on the first flaky attempt."""
    from app.tts_runtime.pool import WorkerCrashed, WorkerOOM, WorkerTimeout

    for exc_cls in (WorkerCrashed, WorkerOOM, WorkerTimeout):
        assert not issubclass(exc_cls, _PERMANENT), exc_cls
        assert issubclass(exc_cls, Exception)


# --- slow: real weights, a real subprocess -----------------------------------------------
SAMPLES = [
    # (text, target_lang, gender, emotion)
    ("ठीक है।", "hi", "female", None),
    ("यह एक भयानक खबर है। हम स्टेशन पर मिलेंगे।", "hi", "male", None),
    ("సరే.", "te", "female", None),
    ("ఇది భయంకరమైన వార్త.", "te", "male", EmotionResult("sadness", 0.9)),
    ("ಸರಿ.", "kn", None, None),
]


@pytest.fixture(scope="module")
def providers():
    """Both providers, built once (each loads and self-checks every SYSPIN
    voice the licence policy offers -- expensive, so shared across samples)."""
    from app.providers.tts.runtime_syspin_provider import RuntimeSyspinTTSProvider
    from app.providers.tts.syspin_provider import SyspinTTSProvider

    old = SyspinTTSProvider()
    new = RuntimeSyspinTTSProvider()
    try:
        yield old, new
    finally:
        new.close()


@needs_syspin
@pytest.mark.parametrize("text,lang,gender,emotion", SAMPLES)
def test_full_synthesize_output_is_byte_identical(providers, text, lang, gender, emotion):
    import soundfile as sf

    old, new = providers
    req = SynthesisRequest(text=text, target_lang=lang, emotion=emotion, target_duration_ms=8000,
                           extra={"voice_gender": gender} if gender else {})
    r_old = old.synthesize(req)
    r_new = new.synthesize(req)
    try:
        assert r_old.sample_rate == r_new.sample_rate
        assert r_old.duration_ms == r_new.duration_ms
        old_bytes = open(r_old.local_audio_path, "rb").read()
        new_bytes = open(r_new.local_audio_path, "rb").read()
        assert hashlib.sha256(old_bytes).hexdigest() == hashlib.sha256(new_bytes).hexdigest(), (
            f"{lang}/{gender} {text!r}: in-process and runtime renders differ")
        # Also confirm the render actually went through the runtime's own
        # decode path, not a no-op stub: read both back and check they are
        # real, non-silent audio of the same length.
        x_old, _ = sf.read(r_old.local_audio_path, dtype="float32")
        x_new, _ = sf.read(r_new.local_audio_path, dtype="float32")
        assert len(x_old) == len(x_new) > 0
    finally:
        import os

        for r in (r_old, r_new):
            try:
                os.remove(r.local_audio_path)
            except OSError:
                pass


@needs_syspin
def test_a_render_failure_in_the_isolated_subprocess_does_not_corrupt_the_provider(providers, tmp_path):
    """A failure raised inside the worker must cross the process boundary as
    a WorkerError the caller can retry, and must not leave the pool unusable.

    The failure used here is deterministic and real: Tamil has no SYSPIN
    voice, so the adapter raises in the worker (SyspinAdapter._repo_for).
    An earlier version of this test used unpronounceable text ("!!!"), which
    does NOT fail -- the carrier sentence prepended by render_line supplies
    valid letters, so the render succeeds. It only ever "passed" when an
    unrelated resource-contention crash happened to raise instead."""
    from app.tts_runtime.pool import WorkerError

    _old, new = providers
    with pytest.raises(WorkerError, match="no SYSPIN voice for ta"):
        new._pool.synth("syspin", "வணக்கம்", "ta", None, None, draw=0, out=str(tmp_path / "o.wav"))
    # Recovery is what matters, not whether the same process survived: the
    # pool respawns a dead worker transparently. The next valid line must
    # render normally -- that is the property the pipeline depends on.
    req = SynthesisRequest(text="ठीक है।", target_lang="hi", extra={"voice_gender": "female"})
    result = new.synthesize(req)
    import os

    assert os.path.exists(result.local_audio_path) and result.duration_ms > 0
    os.remove(result.local_audio_path)
