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
from app.providers.tts.common import render_stable


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

    def render(self, text: str, draw: int = 0) -> np.ndarray:
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


# --- short lines: the duration head is sampled, not trusted once -------------
def test_a_short_line_takes_the_median_of_several_draws():
    """VITS draws each token's duration, and on a one-word line the draws
    spread 2.6x. The run that drew 0.21s for a word that averages 0.45s was
    read back as a different word (CONTRACTS.md #7), so short lines are drawn
    a few times and the middle one kept."""
    from app.providers.tts.common import trim_silence

    drawn = [0.21, 0.45, 0.83]

    def draw(i):
        return _tone(drawn[i], 22050)

    out = render_stable(draw, "abc", 22050)
    assert abs(len(out) / 22050 - len(trim_silence(_tone(0.45, 22050), 22050)) / 22050) < 1e-6


def test_a_long_line_is_rendered_once():
    """Long lines are stable draw to draw and are the expensive ones."""
    calls = []

    def draw(i):
        calls.append(i)
        return _tone(3.0, 22050)

    render_stable(draw, "a line well past the short-line threshold", 22050)
    assert calls == [0]


def test_the_same_line_always_renders_the_same_way():
    """/regenerate re-renders one unchanged segment; it must not come back
    sounding different, and each draw of a short line must be a *different*
    sample. MMS seeded per text -- SYSPIN did not, until it did."""
    import torch

    from app.providers.tts.syspin_provider import SyspinVoice

    voice = SyspinVoice.__new__(SyspinVoice)
    voice.repo_id = "fake/syspin"
    voice.letters = set("abc ")
    voice.tokenizer = type("T", (), {"text_to_ids": staticmethod(lambda t: [1, 2, 3])})()
    # Draws its output from the global RNG, so the seed decides the waveform.
    voice.net = lambda ids: torch.randn(1, 1, 2048)

    first = voice.render("abc")
    assert np.array_equal(first, voice.render("abc")), "same text must re-render identically"
    assert not np.array_equal(first, voice.render("abc", draw=1)), "each draw must be its own sample"
    assert not np.array_equal(first, voice.render("cba")), "different text, different render"
