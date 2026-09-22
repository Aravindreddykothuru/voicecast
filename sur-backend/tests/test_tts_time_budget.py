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
import pathlib

import numpy as np
import pytest

from app.config import get_settings
from app.pipeline.timeline import MAX_TEMPO, TimelineSlot, plan_timeline
from app.providers.base import EmotionResult, SynthesisRequest
from app.providers.tts.common import render_stable

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


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


def test_syspin_cuts_each_language_with_its_own_carrier_and_merge_flag(syspin, monkeypatch):
    """The closure merge only helps if the provider actually asks for it --
    per language, since it is unsafe for Telugu's carrier."""
    from app.capabilities import require_language
    from app.providers.tts import syspin_provider

    provider, _ = syspin
    seen = []

    def fake_render_line(render, text, sr, carrier=None, merge_closures=False):
        seen.append((carrier, merge_closures))
        return _tone(0.5, sr)

    monkeypatch.setattr(syspin_provider, "render_line", fake_render_line)
    for code in ("bn", "te"):
        provider.synthesize(SynthesisRequest(text="x", target_lang=code, target_duration_ms=5000))
    assert seen == [(require_language("bn").tts_carrier, True), (require_language("te").tts_carrier, False)]


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


# --- a short line is spoken after a carrier and cut back out ----------------
def _carrier_render(sr=22050):
    """A fake engine: a carrier utterance, a pause, then the line."""
    calls = []

    def render(text, draw=0):
        calls.append(text)
        if text.startswith("CARRIER"):
            return np.concatenate([_tone(0.9, sr), np.zeros(int(0.30 * sr), dtype="float32"),
                                   _tone(0.40, sr)])
        return _tone(0.40, sr)

    return render, calls


def test_a_short_line_is_rendered_after_the_carrier_and_excised():
    """SYSPIN renders a lone short word as a different word (issue #5): 8 of 8
    one-word lines were misread rendered alone, 8 of 8 read back correctly
    rendered after a carrier and cut out. The line must therefore never be
    rendered on its own when a carrier exists for the language."""
    from app.providers.tts.common import render_line

    sr = 22050
    render, calls = _carrier_render(sr)
    out = render_line(render, "abc", sr, carrier="CARRIER SENTENCE")
    assert calls, "nothing was rendered"
    assert all(c.startswith("CARRIER") for c in calls), f"line rendered bare: {calls}"
    # The clip that ships is the line, not the carrier with it.
    assert 0.30 < len(out) / sr < 0.70, f"expected just the line, got {len(out)/sr:.2f}s"


def test_a_long_line_is_not_given_a_carrier():
    """Long lines never had the defect and are the expensive ones."""
    from app.providers.tts.common import render_line

    sr = 22050
    render, calls = _carrier_render(sr)
    render_line(render, "a line comfortably past the short-line threshold", sr, carrier="CARRIER")
    assert calls == ["a line comfortably past the short-line threshold"]


def test_a_language_with_no_carrier_renders_the_line_as_before():
    from app.providers.tts.common import render_line

    sr = 22050
    render, calls = _carrier_render(sr)
    render_line(render, "abc", sr, carrier=None)
    assert calls == ["abc", "abc", "abc"], calls  # three draws, none scaffolded


def test_a_carrier_render_with_no_pause_falls_back_to_the_bare_line():
    """Better the old behaviour than a mis-cut clip."""
    from app.providers.tts.common import render_line

    sr = 22050
    calls = []

    def render(text, draw=0):
        calls.append(text)
        return _tone(1.4, sr)  # one continuous utterance: no pause to cut at

    out = render_line(render, "abc", sr, carrier="CARRIER")
    assert any(not c.startswith("CARRIER") for c in calls), "never fell back"
    assert len(out) / sr < 1.5


# --- the cut must not start mid-word at a stop closure ----------------------
def _closure_signal(sr=22050, frag1_s=0.20, closure_s=0.16, frag2_s=0.23, carrier_s=1.0, pause_s=0.30):
    """carrier | pause | word part | stop-closure silence | word part."""
    rng = np.random.default_rng(0)
    noise = lambda s: (10 ** (-70 / 20)) * rng.standard_normal(int(s * sr))  # noqa: E731
    return np.concatenate([_tone(carrier_s, sr), noise(pause_s), _tone(frag1_s, sr), noise(closure_s),
                           _tone(frag2_s, sr), noise(0.15)]).astype("float32")


def test_the_cut_keeps_the_whole_word_across_a_stop_closure():
    """A stop consonant's closure is a silence inside the word. The cut used
    to take it for the pause before the line: 92 of 814 renders began
    mid-word, and "সত্যিই" (read back as "তি") passed in 0 of 440 draws."""
    from app.providers.tts.common import tail_after_pause

    sr = 22050
    sig = _closure_signal(sr)
    old = tail_after_pause(sig, sr)
    new = tail_after_pause(sig, sr, merge_closures=True)
    word_s = 0.20 + 0.16 + 0.23
    assert len(old) / sr < word_s, "precondition: the old cut is shorter than the word it should hold"
    assert len(new) - len(old) > int(0.25 * sr), "the merged cut must add the first part back"
    assert len(new) / sr < 0.95, "and must not reach into the 1.0 s carrier"


def test_the_merge_stops_at_the_carrier_even_after_a_short_pause():
    """The guard: a short gap is only a closure if the speech before it is a
    word fragment. Here the carrier is two segments and its final one
    (0.5 s) sits right before a short pause -- without the fragment guard the
    walk back would take the carrier's last words into the clip."""
    from app.providers.tts.common import tail_after_pause

    sr = 22050
    rng = np.random.default_rng(1)
    noise = lambda s: (10 ** (-70 / 20)) * rng.standard_normal(int(s * sr))  # noqa: E731
    sig = np.concatenate([_tone(0.8, sr), noise(0.25), _tone(0.5, sr), noise(0.16),
                          _tone(0.3, sr), noise(0.15)]).astype("float32")
    import librosa
    iv = librosa.effects.split(sig, top_db=30)
    assert len(iv) == 3, f"precondition: three segments, got {len(iv)}"
    gap_ms = 1000 * (iv[2][0] - iv[1][1]) / sr
    assert gap_ms < 140, f"precondition: the pause before the line reads as short ({gap_ms:.0f} ms)"
    new = tail_after_pause(sig, sr, merge_closures=True)
    assert new is not None and len(new) / sr < 0.6, f"carrier leaked into the clip ({len(new) / sr:.2f}s)"


def test_the_real_satyii_render_is_cut_at_the_boundary_not_the_closure():
    """Committed render of "<carrier> সত্যিই?" (draw 0). Its silences are
    [209, 116, 395, 93] ms; the 395 ms one is the boundary, the 93 ms one is
    the closure of "ত্". The old cut kept 335 ms ("তিই"); the whole word is
    232 + 93 + 232 ms."""
    sf = pytest.importorskip("soundfile")
    from app.providers.tts.common import tail_after_pause

    wav, sr = sf.read(FIXTURES_DIR / "closure_bn_satyii_carrier_render.wav", dtype="float32")
    old = tail_after_pause(wav, sr)
    new = tail_after_pause(wav, sr, merge_closures=True)
    assert abs(len(old) / sr - 0.335) < 0.02, "fixture no longer reproduces the mid-word cut"
    assert 0.60 <= len(new) / sr <= 0.72, f"expected the whole word plus pads, got {len(new) / sr:.3f}s"


def test_merge_off_is_exactly_the_old_cut():
    from app.providers.tts.common import tail_after_pause

    sr = 22050
    sig = _closure_signal(sr)
    assert np.array_equal(tail_after_pause(sig, sr), tail_after_pause(sig, sr, merge_closures=False))


def test_the_merge_is_on_only_where_it_was_measured_safe():
    """Measured per carrier against forced-alignment ground truth, 814
    renders: no carrier leak for hi, kn, mr, bn; Telugu's carrier ends on a
    short word the merge swept into the clip 13 times in 407."""
    from app.capabilities import SUPPORTED_LANGUAGES

    flags = {lang.code: lang.tts_carrier_merge_closures for lang in SUPPORTED_LANGUAGES if lang.tts_carrier}
    assert flags == {"te": False, "hi": True, "kn": True, "mr": True, "bn": True}
    assert not any(lang.tts_carrier_merge_closures for lang in SUPPORTED_LANGUAGES if not lang.tts_carrier)


def test_render_line_passes_the_merge_through():
    from app.providers.tts.common import render_line

    sr = 22050
    sig = _closure_signal(sr)

    def render(text, draw=0):
        return sig if text.startswith("CARRIER") else _tone(0.4, sr)

    off = render_line(render, "abc", sr, carrier="CARRIER")
    on = render_line(render, "abc", sr, carrier="CARRIER", merge_closures=True)
    assert len(on) > len(off) + int(0.25 * sr)
