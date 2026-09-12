"""Indic TTS: MMS-TTS (VITS) speaks the words; CosyVoice2 voice conversion
optionally re-voices them as the original speaker.

Why not CosyVoice2 end to end, which is what TTS_PROVIDER=real used to mean:
CosyVoice2's text model is trained on Chinese/English/Japanese/Korean and
has never seen an Indic script. Measured on this project's CPU worker, one
3-second Telugu sentence took 30+ CPU-minutes and came back as ~21s of
babble -- its LLM never emits a stop token for text it can't read, so it
generates to its length cap. MMS-TTS has a checkpoint per Indic language;
the same sentence renders in ~5s and Whisper large-v3 transcribes it back
at CER 0.09 with Telugu auto-detected at 98% confidence.

Emotion: MMS has no emotion conditioning, so the detected register is
carried by prosody -- speaking rate, energy and VITS' own variation scale --
weighted by the classifier's confidence. That is honestly less than a model
conditioned on emotion, and is documented as such rather than dressed up as
more.

No pitch shift, deliberately. It was here (librosa phase vocoder, up to
+/-1.5 semitones) until it was measured: on "అందరికీ శుభోదయం." Whisper CER
went 0.21 -> 0.36 with the happiness shift and back to 0.21 with only the
shift removed; combined with rate squeezing it reached 0.86. The vocoder
smears formants, and on short Indic syllables that costs the words.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import unicodedata
from dataclasses import dataclass

from app.capabilities import require_language
from app.config import get_settings
from app.providers.base import EmotionResult, SynthesisRequest, SynthesisResult, TTSProvider
from app.providers.registry import ProviderNotInstalledError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prosody:
    rate: float = 1.0          # VITS speaking_rate (>1 = faster)
    gain_db: float = 0.0       # relative to the normalized level
    noise_delta: float = 0.0   # added to VITS noise_scale (more/less variation)


# Direction per label follows the acoustic correlates of each emotion (rate,
# intensity, variability); magnitudes are deliberately modest so a
# misclassification is a slightly odd read, not a caricature. No pitch
# component -- see the module docstring.
EMOTION_PROSODY: dict[str, Prosody] = {
    "neutral": Prosody(),
    "happiness": Prosody(rate=1.06, gain_db=1.5, noise_delta=0.08),
    "anger": Prosody(rate=1.08, gain_db=3.0, noise_delta=0.13),
    "sadness": Prosody(rate=0.90, gain_db=-2.0, noise_delta=-0.12),
    "fear": Prosody(rate=1.08, gain_db=-1.0, noise_delta=0.08),
    "surprise": Prosody(rate=1.04, gain_db=1.5, noise_delta=0.13),
}

# Speaking-rate ceiling when fitting a line to the time the picture allows.
# app/pipeline/timeline.py can speed a clip up a little more at mux time;
# together they stay short of speech that sounds sped up.
MAX_FIT_RATE = 1.25
TARGET_RMS_DBFS = -20.0


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


class MMSTTSProvider(TTSProvider):
    def __init__(self) -> None:
        try:
            import librosa  # noqa: F401
            import soundfile  # noqa: F401
            import torch  # noqa: F401
            from transformers import AutoTokenizer, VitsModel  # noqa: F401
        except ImportError as e:
            raise ProviderNotInstalledError("MMSTTSProvider", "torch, transformers, librosa, soundfile") from e

        self._settings = get_settings()
        self._voices: dict[str, tuple] = {}
        self._converter = None
        if self._settings.tts_voice_clone:
            from app.providers.tts.cosyvoice_vc import CosyVoiceVoiceConverter

            self._converter = CosyVoiceVoiceConverter()
        self._self_check(self._settings.default_target_language)

    # -- model management -------------------------------------------------
    def _voice(self, lang_code: str):
        if lang_code in self._voices:
            return self._voices[lang_code]
        lang = require_language(lang_code)
        if not lang.mms_tts:
            raise ValueError(f"no TTS voice is configured for {lang.name} ({lang.code})")

        from transformers import AutoTokenizer, VitsModel

        from app.providers.loading import load_hf_model

        name = f"{self._settings.tts_mms_model_prefix}{lang.mms_tts}"
        tokenizer = AutoTokenizer.from_pretrained(name)
        # Fail-loud load (CONTRACTS.md #1): VITS reports its weight-norm
        # layers under the renamed parametrization keys, which the loader
        # accepts only pairwise against the checkpoint's own weight_g/weight_v.
        model = load_hf_model(VitsModel, name)
        model.eval()
        self._voices[lang_code] = (tokenizer, model)
        logger.info("TTS: loaded %s for %s", name, lang.name)
        return self._voices[lang_code]

    def _self_check(self, lang_code: str) -> None:
        """Render a fixed probe and refuse to boot on silence, NaNs or a
        blown-out signal -- the audible shapes of a broken checkpoint."""
        import numpy as np

        tokenizer, _ = self._voice(lang_code)
        letters = [t for t in tokenizer.get_vocab() if len(t) == 1 and unicodedata.category(t)[0] == "L"]
        probe = " ".join("".join(letters[i:i + 4]) for i in range(0, min(len(letters), 16), 4))
        wav, sr = self._render(lang_code, probe, Prosody())
        rms_db = 20 * np.log10(float(np.sqrt(np.mean(wav ** 2))) + 1e-12)
        if not np.isfinite(wav).all() or len(wav) < sr * 0.2 or rms_db < -50 or float(np.max(np.abs(wav))) > 4.0:
            from app.providers.loading import ModelLoadError

            raise ModelLoadError(
                f"TTS self-check failed for {lang_code}: {len(wav) / sr:.2f}s, "
                f"RMS {rms_db:.1f} dBFS -- the voice produced no usable audio."
            )
        logger.info("TTS self-check passed for %s (%.2fs, RMS %.1f dBFS)", lang_code, len(wav) / sr, rms_db)

    # -- rendering --------------------------------------------------------
    def _render(self, lang_code: str, text: str, prosody: Prosody, *, report_dropped: bool = True):
        import torch

        tokenizer, model = self._voice(lang_code)
        vocab = tokenizer.get_vocab()
        lowered = unicodedata.normalize("NFC", text.lower())
        dropped = sorted({c for c in lowered if c not in vocab and unicodedata.category(c)[0] in "LNM"})
        spoken = [c for c in lowered if c in vocab and unicodedata.category(c)[0] in "LNM"]
        if not spoken:
            raise ValueError(f"the {lang_code} voice cannot pronounce any character of {text!r}")
        if dropped and report_dropped:
            # Most often digits: VITS vocabularies are letters only, so "40"
            # is silently skipped by the tokenizer. Logged, never hidden.
            logger.warning("TTS (%s) cannot pronounce and will skip: %s", lang_code, "".join(dropped))

        inputs = tokenizer(text, return_tensors="pt")
        model.speaking_rate = prosody.rate
        model.noise_scale = model.config.noise_scale + prosody.noise_delta
        # Deterministic per text: re-rendering an unchanged line must not
        # change how it sounds.
        torch.manual_seed(int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16))
        with torch.inference_mode():
            wav = model(**inputs).waveform[0].float().numpy()
        return wav, model.config.sampling_rate

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        import soundfile as sf

        text = request.text.strip()
        if not text:
            raise ValueError("TTS was asked to speak an empty string")
        lang = require_language(request.target_lang)
        prosody = prosody_for(request.emotion, self._settings.emotion_confidence_floor)

        wav, sr = self._render(lang.code, text, prosody)
        wav = _trim_silence(wav, sr)

        target = request.target_duration_ms
        if target:
            natural_ms = 1000 * len(wav) / sr
            if natural_ms > target * 1.05:
                fit_rate = min(prosody.rate * natural_ms / target, MAX_FIT_RATE)
                if fit_rate > prosody.rate + 0.02:
                    faster = Prosody(fit_rate, prosody.gain_db, prosody.noise_delta)
                    wav, sr = self._render(lang.code, text, faster, report_dropped=False)
                    wav = _trim_silence(wav, sr)

        if request.voice_reference_path:
            if self._converter is None:
                raise ValueError(
                    "voice cloning was requested but TTS_VOICE_CLONE is off on this worker"
                )
            wav, sr = self._converter.convert(wav, sr, request.voice_reference_path)
            wav = _trim_silence(wav, sr)

        wav = _normalize(wav, prosody.gain_db)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # release our handle first -- see ffmpeg_utils.extract_audio_slice
        sf.write(path, wav, sr, subtype="PCM_16")
        return SynthesisResult(local_audio_path=path, duration_ms=int(1000 * len(wav) / sr), sample_rate=sr)


def _trim_silence(wav, sr: int, top_db: float = 40.0):
    """Leading silence would land the speech after the segment's start_ms --
    a sync error introduced by the renderer, not the source."""
    import librosa

    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    pad = int(0.03 * sr)
    return trimmed if len(trimmed) > pad else wav


def _normalize(wav, gain_db: float):
    """Consistent loudness across segments (and across voices, after VC),
    then the emotion's relative gain, with a peak limit."""
    import numpy as np

    rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-12
    out = wav * (10 ** ((TARGET_RMS_DBFS + gain_db) / 20) / rms)
    peak = float(np.max(np.abs(out)))
    if peak > 0.97:
        out = out * (0.97 / peak)
    return out.astype("float32")
