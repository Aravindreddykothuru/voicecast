"""MMS-TTS (facebook/mms-tts-*, VITS) -- RESEARCH / NON-COMMERCIAL ENGINE.

Weights are CC-BY-NC-4.0. The registry refuses this engine while
TTS_REQUIRE_COMMERCIAL_LICENSE is true; use TTS_ENGINE=syspin for anything
commercial. It stays because it covers all twelve target languages and has
the highest measured intelligibility on Telugu (CONTRACTS.md #7).

Why not CosyVoice2, which is what TTS_PROVIDER=real once meant: its text
model has never seen an Indic script. One 3-second Telugu sentence took 30+
CPU-minutes and came back as ~21s of babble (Whisper CER 1.56).

Emotion hooks this model genuinely exposes (verified on VitsModel):
speaking_rate and noise_scale. No pitch or emotion conditioning exists, so
prosody is rate, variation and energy -- see app/providers/tts/common.py,
including why there is deliberately no post-hoc pitch shift.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import unicodedata

from app.capabilities import require_language
from app.config import get_settings
from app.providers.base import SynthesisRequest, SynthesisResult, TTSProvider
from app.providers.registry import ProviderNotInstalledError
from app.providers.tts.common import EMOTION_PROSODY, Prosody, fits_window, normalize, prosody_for, trim_silence

logger = logging.getLogger(__name__)

# Re-exported: tests and evaluation scripts import these from here.
__all__ = ["EMOTION_PROSODY", "MMSTTSProvider", "Prosody", "prosody_for"]
_trim_silence = trim_silence
_normalize = normalize


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
            from app.providers.tts.voice_clone import get_voice_converter

            self._converter = get_voice_converter()
        self._self_check(self._settings.default_target_language)

    # -- model management -------------------------------------------------
    def _voice(self, lang_code: str):
        if lang_code in self._voices:
            return self._voices[lang_code]
        lang = require_language(lang_code)
        voice = next((v for v in lang.voices if v.engine == "mms"), None)
        if voice is None:
            raise ValueError(f"no MMS voice is configured for {lang.name} ({lang.code})")

        from transformers import AutoTokenizer, VitsModel

        from app.providers.loading import load_hf_model

        tokenizer = AutoTokenizer.from_pretrained(voice.model)
        # Fail-loud load (CONTRACTS.md #1): VITS reports its weight-norm
        # layers under the renamed parametrization keys, which the loader
        # accepts only pairwise against the checkpoint's own weight_g/weight_v.
        model = load_hf_model(VitsModel, voice.model)
        model.eval()
        self._voices[lang_code] = (tokenizer, model)
        logger.info("TTS: loaded %s for %s (%s)", voice.model, lang.name, voice.license)
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
        wav = trim_silence(wav, sr)

        # No time fitting here: mux_export (timeline.plan_timeline) is the one
        # place a clip is sped up to its window, within one MAX_TEMPO budget
        # (CONTRACTS.md #7). A line that won't fit anyway is re-rendered at
        # neutral rate, so the emotion's rate doesn't compound with the mux's.
        if abs(prosody.rate - 1.0) > 0.01 and not fits_window(len(wav) / sr, request.target_duration_ms):
            neutral_rate = Prosody(1.0, prosody.gain_db, prosody.noise_delta)
            wav, sr = self._render(lang.code, text, neutral_rate, report_dropped=False)
            wav = trim_silence(wav, sr)

        if request.voice_reference_path:
            if self._converter is None:
                raise ValueError("voice cloning was requested but TTS_VOICE_CLONE is off on this worker")
            wav, sr = self._converter.convert(wav, sr, request.voice_reference_path, voice_key=f"mms:{lang.code}")
            wav = trim_silence(wav, sr)

        wav = normalize(wav, prosody.gain_db)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # release our handle first -- see ffmpeg_utils.extract_audio_slice
        sf.write(path, wav, sr, subtype="PCM_16")
        return SynthesisResult(local_audio_path=path, duration_ms=int(1000 * len(wav) / sr), sample_rate=sr)
