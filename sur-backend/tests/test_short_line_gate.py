"""The intelligibility gate must not fail a dub for a clip too short to score.

`scripts/e2e_dub.py --asr-check` used to hold every line to a CER bar,
including one-word interjections. On the 2026-09-20 run that failed the dub
on line 7, "సరే." (*okay*) -- 327ms with silence on both sides -- which
Whisper transcribed as "క్వే".

The audio was fine. The proof is the pair of fixtures here: the same word,
same voice, same renderer, read back correctly as "సరే" when it sits inside a
carrier sentence and as nonsense when it stands alone. Measured across fifteen
correctly-rendered clips, ten of the twelve under 1100ms failed the 0.35 bar
while every clip from 1207ms up passed -- so below ASR_FLOOR_MS a line is
checked for presence instead of transcribed, and above it nothing changes.

These tests exist so that floor cannot quietly drop back down.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
# Measured from the 2026-09-20 run and its control, at the gate's own 16kHz.
SARE_ISOLATED_MS = 327
SARE_IN_CARRIER_MS = 1997
SARE_RMS_DB = -18.7
SARE_VOICED_FRACTION = 0.86


def _gate():
    """scripts/ is not a package; load the gate by path."""
    path = pathlib.Path(__file__).parent.parent / "scripts" / "e2e_dub.py"
    spec = importlib.util.spec_from_file_location("e2e_dub", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_one_word_interjection_is_not_transcribed_at_all():
    """327ms of correct speech scored CER 1.0. Nothing that short is asked."""
    gate = _gate()
    assert not gate.scores_by_asr(SARE_ISOLATED_MS)


def test_the_same_word_in_a_carrier_sentence_is_still_transcribed():
    """The rule is about clip length, not about the word -- the control reads
    back correctly and stays on the ASR path."""
    gate = _gate()
    assert gate.scores_by_asr(SARE_IN_CARRIER_MS)


def test_lines_that_score_well_today_keep_being_scored():
    """The run's two-word lines measured 1207ms (CER 0.083) and 1217ms
    (0.214). Raising the floor past them would buy a green gate by measuring
    less, which is the failure mode opposite to the one being fixed."""
    gate = _gate()
    assert gate.scores_by_asr(1207)
    assert gate.scores_by_asr(1217)


def test_the_interjection_passes_the_presence_check_it_is_held_to():
    """It is not scored, but it is not unchecked either."""
    gate = _gate()
    assert gate.presence_verdict(SARE_RMS_DB, SARE_VOICED_FRACTION)


@pytest.mark.parametrize("rms_db, voiced, why", [
    (-240.0, 0.00, "silence: the clip never made it into the mux"),
    (-20.0, 0.00, "white noise: audible, but not speech"),
    (-26.6, 0.00, "a click: audible, but not speech"),
    (-50.0, 0.80, "speech so quiet it will not be heard"),
])
def test_presence_still_catches_a_line_that_is_not_there(rms_db, voiced, why):
    gate = _gate()
    assert not gate.presence_verdict(rms_db, voiced), why


def test_the_real_clip_measures_as_speech():
    """End to end on the committed audio: the clip from the dub measures as
    ordinary speech, indistinguishable from the lines that do get scored
    (voiced 0.76-0.92 across that run)."""
    pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    import soundfile as sf

    gate = _gate()
    wav, sr = sf.read(FIXTURES / "short_line_sare_isolated.wav", dtype="float32")
    assert not gate.scores_by_asr(round(1000 * len(wav) / sr))
    measured = gate.measure_presence(wav, sr)
    assert measured["voiced_fraction"] >= gate.PRESENCE_MIN_VOICED
    assert measured["rms_db"] > gate.PRESENCE_MIN_RMS_DB
    assert gate.presence_verdict(**measured)


def test_silence_the_same_length_as_the_clip_does_not_pass():
    pytest.importorskip("librosa")
    import numpy as np

    gate = _gate()
    sr = 22050
    silence = np.zeros(int(sr * SARE_ISOLATED_MS / 1000), dtype="float32")
    assert not gate.presence_verdict(**gate.measure_presence(silence, sr))
