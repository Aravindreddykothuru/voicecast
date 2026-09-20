"""Voice cloning is gated by a budget enforced in code (CONTRACTS.md #7).

Cloning used to be offered however slow it was (12.5 CPU-minutes for 30s of
speech) and however much it garbled the dub. A converter that is too slow,
changes the timing, or loses the speech envelope must now stop the TTS
worker from starting instead of failing -- or quietly degrading -- jobs.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.providers.tts import voice_clone
from app.providers.tts.voice_clone import VoiceCloneBudgetError, check_budget


class _Identity:
    def __init__(self):
        self.calls = []

    def convert(self, wav, sr, reference_path, voice_key=None):
        self.calls.append((len(wav), voice_key))
        return np.asarray(wav, dtype="float32"), sr


class _Stretching(_Identity):
    def convert(self, wav, sr, reference_path, voice_key=None):
        return np.repeat(np.asarray(wav, dtype="float32"), 2), sr


class _Noise(_Identity):
    def convert(self, wav, sr, reference_path, voice_key=None):
        return (0.1 * np.random.default_rng(0).standard_normal(len(wav))).astype("float32"), sr


def _fake_clock(step_seconds: float):
    ticks = itertools.count()
    return lambda: next(ticks) * step_seconds


def test_a_content_preserving_fast_converter_passes():
    converter = _Identity()
    check_budget(converter, clock=_fake_clock(0.3))  # 0.3s for 3s of audio: RTF 0.1
    # One untimed warm-up on a short clip, then the timed 3-second probe.
    assert [key for _, key in converter.calls] == ["__budget_warmup__", "__budget_check__"]
    assert converter.calls[0][0] < converter.calls[1][0]


def test_a_converter_slower_than_the_rtf_budget_is_refused(monkeypatch):
    monkeypatch.setenv("TTS_VOICE_CLONE_MAX_RTF", "1.0")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        with pytest.raises(VoiceCloneBudgetError, match=r"converts at 3\.33x real time, budget is 1\.00x"):
            check_budget(_Identity(), clock=_fake_clock(10.0))
    finally:
        get_settings.cache_clear()


def test_a_transient_slow_run_is_retimed_rather_than_refused(monkeypatch):
    """Contention only ever adds time, so the fastest run is the converter's
    speed. One slow sample must not refuse the worker (it did: the check read
    4.65x for a 2.2x converter when both workers booted together)."""
    monkeypatch.setenv("TTS_VOICE_CLONE_MAX_RTF", "4.0")
    from app.config import get_settings

    get_settings.cache_clear()
    # 30s for the first timed run (RTF 10), 3s for the second (RTF 1.0).
    steps = iter([0.0, 30.0, 30.0, 33.0])
    try:
        check_budget(_Identity(), clock=lambda: next(steps))
    finally:
        get_settings.cache_clear()


def test_a_converter_slow_on_every_attempt_is_still_refused(monkeypatch):
    monkeypatch.setenv("TTS_VOICE_CLONE_MAX_RTF", "4.0")
    from app.config import get_settings

    get_settings.cache_clear()
    converter = _Identity()
    try:
        with pytest.raises(VoiceCloneBudgetError, match=r"converts at 10\.00x real time"):
            check_budget(converter, clock=_fake_clock(30.0))
    finally:
        get_settings.cache_clear()
    # Warm-up plus one timed run per attempt -- it does not retry forever.
    assert len(converter.calls) == 4


def test_a_converter_that_changes_timing_is_refused():
    with pytest.raises(VoiceCloneBudgetError, match="changes duration"):
        check_budget(_Stretching(), clock=_fake_clock(0.1))


def test_a_converter_that_loses_the_speech_envelope_is_refused():
    with pytest.raises(VoiceCloneBudgetError, match="energy envelope"):
        check_budget(_Noise(), clock=_fake_clock(0.1))


def test_the_worker_cannot_get_a_converter_that_fails_its_budget(monkeypatch):
    monkeypatch.setattr(voice_clone, "OpenVoiceConverter", _Noise)
    monkeypatch.setattr(voice_clone.get_settings(), "tts_voice_clone_engine", "openvoice")
    with pytest.raises(VoiceCloneBudgetError):
        voice_clone.get_voice_converter()
