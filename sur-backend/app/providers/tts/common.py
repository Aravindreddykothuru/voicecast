"""Engine-independent TTS post-processing: emotion prosody, silence trimming,
loudness, and speaker voice selection.

Shared by every TTS engine so switching TTS_ENGINE changes the voice, not the
behaviour around it. The measurements behind each choice are in
CONTRACTS.md #7.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.providers.base import EmotionResult

TARGET_RMS_DBFS = -20.0


@dataclass(frozen=True)
class Prosody:
    rate: float = 1.0          # speaking rate (>1 = faster)
    gain_db: float = 0.0       # relative to the normalized level
    noise_delta: float = 0.0   # engines with a variation knob (VITS noise_scale) add this


# Direction per label follows the acoustic correlates of each emotion (rate,
# intensity, variability); magnitudes are deliberately modest so a
# misclassification is a slightly odd read, not a caricature. No post-hoc
# pitch shift: measured, a phase-vocoder shift cost intelligibility on short
# lines (Whisper CER 0.21 -> 0.36).
EMOTION_PROSODY: dict[str, Prosody] = {
    "neutral": Prosody(),
    "happiness": Prosody(rate=1.06, gain_db=1.5, noise_delta=0.08),
    "anger": Prosody(rate=1.08, gain_db=3.0, noise_delta=0.13),
    "sadness": Prosody(rate=0.90, gain_db=-2.0, noise_delta=-0.12),
    "fear": Prosody(rate=1.08, gain_db=-1.0, noise_delta=0.08),
    "surprise": Prosody(rate=1.04, gain_db=1.5, noise_delta=0.13),
}


def prosody_for(emotion: EmotionResult | None, confidence_floor: float) -> Prosody:
    """Blend from neutral toward the label's prosody by classifier confidence.

    Below the confidence floor the UI shows the label as "uncertain"
    (/api/capabilities), so rendering it would contradict what the reviewer
    sees; it renders neutral. An unknown label raises rather than silently
    reading as neutral."""
    if emotion is None:
        return Prosody()
    target = EMOTION_PROSODY.get(emotion.label)
    if target is None:
        raise ValueError(f"no prosody mapping for emotion label {emotion.label!r}")
    span = max(1.0 - confidence_floor, 1e-6)
    w = min(max((emotion.score - confidence_floor) / span, 0.0), 1.0)
    return Prosody(
        rate=1.0 + (target.rate - 1.0) * w,
        gain_db=target.gain_db * w,
        noise_delta=target.noise_delta * w,
    )


def fits_window(natural_seconds: float, window_ms: int | None) -> bool:
    """Whether a clip of this natural length fits the time before the next
    line. Engines use it only to decide whether an emotion's rate change is
    worth applying; the actual fitting is mux_export's (plan_timeline)."""
    return not window_ms or 1000 * natural_seconds <= window_ms * 1.05


def trim_silence(wav, sr: int, top_db: float = 40.0):
    """Leading silence would land the speech after the segment's start_ms --
    a sync error introduced by the renderer, not the source."""
    import librosa

    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    pad = int(0.03 * sr)
    return trimmed if len(trimmed) > pad else wav


# VITS predicts each token's duration stochastically, and the relative spread
# explodes as a line gets shorter: ten draws of the one-word Telugu line
# "sare" measured 0.33-0.84s (x2.6), while a 19-character line stayed within
# x1.18. The end-to-end run drew 0.21s for that word and Whisper read it back
# as a different word entirely (CER 1.33) -- the rushed tail of the
# distribution is unintelligible, not merely brisk. Short lines are therefore
# drawn a few times and the median kept: milliseconds on the cheapest lines in
# a run, and it drops the rushed draw without chasing the drawn-out one. Long
# lines are left alone -- they are stable, and they are the expensive ones.
SHORT_LINE_CHARS = 12
SHORT_LINE_DRAWS = 3


def tail_after_pause(wav, sr: int, min_ms: int = 120, max_ms: int = 2000):
    """The last utterance in a clip, cut at the pause before it, or None.

    Used to recover a short line rendered after a carrier sentence. The
    threshold is a ladder because one fixed value does not find the pause in
    every render: at 35 dB it missed 4 of 40, at 30 dB it missed 2. A segment
    of implausible length is rejected rather than returned, so the caller can
    fall back instead of shipping a mis-cut clip."""
    import librosa

    for top_db in (30, 35, 40, 45, 50):
        intervals = librosa.effects.split(wav, top_db=top_db)
        if len(intervals) < 2:
            continue
        a, b = intervals[-1]
        if min_ms <= 1000 * (b - a) / sr <= max_ms:
            # Generous at the head: a word opening on a vowel fades in below
            # the split threshold, and cutting 30ms tight lost the first
            # syllable of "ఇప్పుడే" (read back as "పుడే").
            return wav[max(a - int(0.08 * sr), 0):min(b + int(0.06 * sr), len(wav))]
    return None


def render_line(render, text: str, sr: int, carrier: str | None = None):
    """One line of speech, trimmed, as the engine should have rendered it.

    A line short enough to be the entire utterance is rendered *after*
    `carrier` and cut back out, because SYSPIN renders a lone short word as a
    different word (issue #5). Measured over 8 one-word Telugu lines, median
    of 3 draws, CTC read-back: 2/8 intelligible rendered alone, 8/8 rendered
    after a carrier and excised. The clip that ships is still only the line,
    at its natural length -- the carrier is scaffolding, never heard.

    `render(text, draw=i)` must be deterministic in `i`."""
    if carrier and len(text) <= SHORT_LINE_CHARS:
        def draw(i):
            whole = render(f"{carrier} {text}", draw=i)
            tail = tail_after_pause(whole, sr)
            # No pause found: fall back to rendering the line by itself
            # rather than shipping a mis-cut carrier.
            return tail if tail is not None else trim_silence(render(text, draw=i), sr)

        # Not trimmed again: the cut already sits on silence either side, and
        # re-trimming it to 40 dB took the soft onset back off (8/9 -> 9/9).
        return render_stable(draw, text, sr, trim=False)
    return render_stable(lambda i: render(text, draw=i), text, sr)


def render_stable(draw, text: str, sr: int, *, trim: bool = True):
    """Render `text` to a clip, resampling the duration head on short lines.

    `draw(i)` renders draw number `i`. It must be deterministic in `i` -- the
    same line has to give the same clip every time, so that re-rendering one
    unchanged segment does not change how it sounds (CONTRACTS.md #7).
    `trim=False` for draws that are already cut to their own boundaries."""
    prep = (lambda w: trim_silence(w, sr)) if trim else (lambda w: w)
    first = prep(draw(0))
    if len(text) > SHORT_LINE_CHARS:
        return first
    clips = sorted([first] + [prep(draw(i)) for i in range(1, SHORT_LINE_DRAWS)], key=len)
    return clips[len(clips) // 2]


def normalize(wav, gain_db: float):
    """Consistent loudness across segments (and across voices, after voice
    conversion), then the emotion's relative gain, with a peak limit."""
    import numpy as np

    rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-12
    out = wav * (10 ** ((TARGET_RMS_DBFS + gain_db) / 20) / rms)
    peak = float(np.max(np.abs(out)))
    if peak > 0.97:
        out = out * (0.97 / peak)
    return out.astype("float32")


def estimate_voice_gender(wav, sr: int) -> str | None:
    """"male"/"female" from median voiced F0, or None when there is too
    little voiced speech to tell. Used only to pick which of an engine's
    voices dubs a speaker -- two speakers in a scene should not collapse
    into one voice."""
    import librosa
    import numpy as np

    f0, voiced, _ = librosa.pyin(wav, fmin=60, fmax=400, sr=sr, frame_length=1024)
    f0 = f0[voiced & np.isfinite(f0)]
    if len(f0) < 20:
        return None
    return "male" if float(np.median(f0)) < 165.0 else "female"
