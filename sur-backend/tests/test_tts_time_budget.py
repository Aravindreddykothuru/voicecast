"""One tempo budget per clip: TTS engines never time-fit, mux_export does.

Both engines used to speed a line up to 1.25x to fit its window, after which
plan_timeline sped the *already fitted* clip up to another 1.35x -- 1.69x in
total, two WSOLA passes, and the QA report's `tempo` showed only the second
factor. Short lines, whose windows are tightest, took the worst of it
(CONTRACTS.md #7). These run the real synthesize() code paths with the model
call faked, so no weights are needed.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from app.config import get_settings
from app.pipeline.timeline import MAX_TEMPO, TimelineSlot, plan_timeline
from app.providers.base import EmotionResult, SynthesisRequest


def _tone(seconds: float, sr: int) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")


def _duration_ms(result) -> int:
    try:
        return result.duration_ms
    finally:
        os.remove(result.local_audio_path)


class _FakeSyspinVoice:
    repo_id = "fake/syspin"

    def render(self, text: str) -> np.ndarray:
        return _tone(1.0, 22050)


@pytest.fixture
def syspin(monkeypatch):
    from app.pipeline import ffmpeg_utils
    from app.providers.tts.syspin_provider import SyspinTTSProvider

    provider = SyspinTTSProvider.__new__(SyspinTTSProvider)
    provider._settings = get_settings()
    provider._voices = {}
    provider._converter = None
    monkeypatch.setattr(provider, "_voice_for", lambda lang, gender: _FakeSyspinVoice())
    stretches: list[float] = []

    def fake_stretch(wav, sr, rate):
        stretches.append(rate)
        return wav[: int(len(wav) / rate)]

    monkeypatch.setattr(ffmpeg_utils, "time_stretch", fake_stretch)
    return provider, stretches


def test_syspin_leaves_a_line_longer_than_its_window_at_natural_length(syspin):
    provider, stretches = syspin
    result = provider.synthesize(SynthesisRequest(text="x", target_lang="te", target_duration_ms=500))
    assert stretches == []
    assert _duration_ms(result) == pytest.approx(1000, abs=5)


def test_syspin_applies_the_emotion_rate_only_when_the_line_fits(syspin):
    provider, stretches = syspin
    happy = EmotionResult("happiness", 1.0)
    provider.synthesize(SynthesisRequest(text="x", target_lang="te", emotion=happy, target_duration_ms=5000))
    assert stretches == [pytest.approx(1.06)]

    stretches.clear()
    sad = EmotionResult("sadness", 1.0)  # 0.90x would make an overrun worse
    provider.synthesize(SynthesisRequest(text="x", target_lang="te", emotion=sad, target_duration_ms=600))
    assert stretches == []


@pytest.fixture
def mms(monkeypatch):
    from app.providers.tts.mms_provider import MMSTTSProvider

    provider = MMSTTSProvider.__new__(MMSTTSProvider)
    provider._settings = get_settings()
    provider._voices = {}
    provider._converter = None
    rates: list[float] = []

    def fake_render(lang_code, text, prosody, *, report_dropped=True):
        rates.append(round(prosody.rate, 4))
        return _tone(1.0 / prosody.rate, 16000), 16000

    monkeypatch.setattr(provider, "_render", fake_render)
    return provider, rates


def test_mms_never_renders_faster_than_the_emotion_rate_to_fit(mms):
    provider, rates = mms
    result = provider.synthesize(SynthesisRequest(text="x", target_lang="te", target_duration_ms=500))
    assert rates == [1.0]
    assert _duration_ms(result) == pytest.approx(1000, abs=5)


def test_mms_drops_the_emotion_rate_for_a_line_that_does_not_fit(mms):
    provider, rates = mms
    sad = EmotionResult("sadness", 1.0)
    provider.synthesize(SynthesisRequest(text="x", target_lang="te", emotion=sad, target_duration_ms=600))
    assert rates == [0.9, 1.0]

    rates.clear()
    happy = EmotionResult("happiness", 1.0)
    provider.synthesize(SynthesisRequest(text="x", target_lang="te", emotion=happy, target_duration_ms=5000))
    assert rates == [1.06]


def test_the_mux_tempo_is_the_whole_speed_up(syspin):
    """With the engine no longer fitting, the planner's tempo is the total
    compression, so capping it at MAX_TEMPO caps the clip."""
    provider, _ = syspin
    clip_ms = _duration_ms(provider.synthesize(SynthesisRequest(text="x", target_lang="te", target_duration_ms=500)))
    [planned] = plan_timeline([TimelineSlot("s", 0, 400, clip_ms)], 560)
    assert planned.tempo == MAX_TEMPO
    assert planned.fitted_ms == pytest.approx(clip_ms / MAX_TEMPO, abs=1)
