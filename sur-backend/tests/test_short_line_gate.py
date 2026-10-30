"""Short lines are verified for *what* was said, not just that sound exists.

Whisper cannot read a clip much under 1.2s: across fifteen clips, ten of the
twelve under 1100ms scored past the 0.35 bar. So below ASR_FLOOR_MS the gate
does not ask Whisper. It asks a frame-synchronous CTC recogniser, which has
no length prior -- on complete Telugu words decoded entirely alone, 360-1060ms,
it was exact on 7 of 8, and it rejects silence, noise, a click and a
substituted word.

The three fixtures here are the experiment that settled what "too short"
means, and they are kept because the answer was counter-intuitive:

  short_line_sare_isolated.wav        the dub's own line 7, 327ms, the
                                      pipeline's standalone one-word render
  short_line_sare_excised_from_carrier.wav
                                      the SAME word cut out of a carrier
                                      sentence, 380ms, no context around it
  short_line_sare_in_carrier.wav      that carrier sentence, 1997ms

Both recognisers read the excised 380ms clip correctly and both misread the
327ms standalone render as "క(్)వే". Same word, same voice, same length,
opposite results -- so the standalone *render* is degraded, and the clip
length was never the reason. A gate that waves short lines through on
"there is speech here" would pass that defect, which is why the CTC check
exists and why the presence check is only the fallback.
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
# CTC CER measured on each fixture against the reference "సరే".
SARE_ISOLATED_CTC_CER = 0.667      # decoded "కవే" -- the degraded render
SARE_EXCISED_CTC_CER = 0.0         # decoded "సరే" -- good audio, 380ms


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


# --- below the Whisper floor the gate reads the words, not just the level ---
def test_a_degraded_short_render_is_rejected_even_though_speech_is_present():
    """The failure this whole check exists for: line 7 measures as perfectly
    ordinary speech (-18.7 dBFS, voiced 0.86) and is still the wrong word."""
    gate = _gate()
    assert gate.presence_verdict(SARE_RMS_DB, SARE_VOICED_FRACTION), "it is audible speech"
    assert not gate.short_line_verdict(SARE_ISOLATED_CTC_CER, SARE_RMS_DB), "but not the right speech"


def test_good_short_audio_passes_below_the_floor():
    """380ms, well under ASR_FLOOR_MS, read back exactly. Short is not the
    problem, so short must not be an automatic failure either."""
    gate = _gate()
    assert not gate.scores_by_asr(380)
    assert gate.short_line_verdict(SARE_EXCISED_CTC_CER, -20.0)


def test_a_correct_but_inaudible_clip_is_rejected():
    """wav2vec2 normalises its input, so a clip at -50 dBFS decodes
    perfectly (measured CER 0.000). Level has to be checked separately."""
    gate = _gate()
    assert not gate.short_line_verdict(0.0, -50.0)


def test_an_unsupported_language_falls_back_rather_than_passing():
    gate = _gate()
    # Tamil has no commercially licensed voice, so it was never validated.
    assert gate.ctc_reader("ta") is None
    assert "te" in gate.SHORT_LINE_CTC_MODELS


def test_the_ctc_reader_agrees_with_the_measurements_this_gate_was_tuned_on():
    """The end-to-end proof, on the committed audio. Needs the CTC weights,
    so it is skipped where they are not available -- the pure-logic tests
    above still pin the thresholds."""
    pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    pytest.importorskip("transformers")
    import librosa
    import soundfile as sf

    gate = _gate()
    read = gate.ctc_reader("te")
    if read is None:
        pytest.skip("CTC weights unavailable")

    def decode(name):
        wav, sr = sf.read(FIXTURES / name, dtype="float32")
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        return read(wav)

    good = gate._cer("సరే", decode("short_line_sare_excised_from_carrier.wav"))
    bad = gate._cer("సరే", decode("short_line_sare_isolated.wav"))
    assert good <= gate.SHORT_LINE_MAX_CER, "380ms of good audio must read back"
    assert bad > gate.SHORT_LINE_MAX_CER, "the degraded render must not"


# --- the same defect, and the same fix, in every covered language ----------
# Rendered through the provider's own path: "alone" is what shipped before,
# "carrier_excised" is the line spoken after a carrier sentence and cut back
# out. Reference text is the pipeline's own translation of an English line.
PER_LANGUAGE = [
    ("kn", "ಏಕೆ", "Why?"),
]
# hi/mr/bn fixtures are kept in tests/fixtures and their before/after numbers
# are in CONTRACTS.md #7, but they are not asserted here: those languages have
# no reader in the gate, so there is nothing to assert them with.


def test_a_language_is_only_read_back_if_it_is_also_carrier_rendered():
    """Readers are a subset of carriers, not a match for them.

    A carrier improves the audio in every language measured, so it is set
    wherever it was measured. A reader only earns its place by passing
    *correct* short lines: hi, mr and bn have carriers but no reader, because
    their readers failed 2-3 of 8 correct lines. Reading a language the
    carrier never touched would be the other way round and is a mistake."""
    from app.capabilities import SUPPORTED_LANGUAGES

    gate = _gate()
    carriers = {lang.code for lang in SUPPORTED_LANGUAGES if lang.tts_carrier}
    readers = set(gate.SHORT_LINE_CTC_MODELS)
    assert readers <= carriers, f"reader without a carrier: {readers - carriers}"
    assert readers == {"te", "kn"}
    assert carriers == {"te", "hi", "kn", "mr", "bn"}


def test_languages_without_a_reader_fall_back_rather_than_pass():
    from app.capabilities import SUPPORTED_LANGUAGES

    gate = _gate()
    uncovered = [l.code for l in SUPPORTED_LANGUAGES if l.code not in gate.SHORT_LINE_CTC_MODELS]
    # Tracked explicitly. as/gu/ml/or/pa/ta have no commercially licensed
    # voice, so they cannot be dubbed on the default configuration and there
    # is no shipping audio to validate a reader against. hi/mr/bn can be
    # dubbed, and their readers were measured and rejected -- they failed
    # 2-3 of 8 correct short lines each.
    assert sorted(uncovered) == ["as", "bn", "gu", "hi", "ml", "mr", "or", "pa", "ta"]
    for code in uncovered:
        assert gate.ctc_reader(code) is None


@pytest.mark.parametrize("code, ref, source", PER_LANGUAGE)
def test_the_carrier_fix_holds_in_each_language(code, ref, source):
    """Before: the line rendered alone. After: rendered after a carrier and
    excised. Same synthesizer, same voice, same line."""
    pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    pytest.importorskip("transformers")
    import librosa
    import soundfile as sf

    gate = _gate()
    read = gate.ctc_reader(code)
    if read is None:
        pytest.skip(f"CTC weights for {code} unavailable")

    def cer_of(name):
        wav, sr = sf.read(FIXTURES / name, dtype="float32")
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        return gate._cer(ref, read(wav))

    before = cer_of(f"short_line_{code}_alone.wav")
    after = cer_of(f"short_line_{code}_carrier_excised.wav")
    assert after <= gate.SHORT_LINE_MAX_CER, f"{code}: the fixed render must read back ({after:.3f})"
    assert before > gate.SHORT_LINE_MAX_CER, f"{code}: the old render must not ({before:.3f})"


def test_every_carrier_records_who_checked_it():
    """A carrier is machine-translated text that drives the engine. It is
    never heard, but it must not be assumed good either -- each one carries
    its review state, and today none has been read by a fluent speaker."""
    from app.capabilities import SUPPORTED_LANGUAGES

    carried = [l for l in SUPPORTED_LANGUAGES if l.tts_carrier]
    assert carried, "no carriers configured"
    for lang in carried:
        assert lang.tts_carrier_review, f"{lang.code} carrier has no review state"
    human = [l.code for l in carried if not l.tts_carrier_review.startswith("machine:")]
    # Update this when a fluent speaker signs one off -- the assertion is
    # here so that "reviewed" cannot quietly become the assumed default.
    assert human == [], f"human-reviewed carriers now exist, update the docs: {human}"
