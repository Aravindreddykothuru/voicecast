"""SYSPIN VITS (IISc / ARTPARK) text-to-speech -- the commercially licensed engine.

Weights: CC-BY-4.0 (commercial use with attribution). Each release is a
TorchScript VITS export plus a character list; text processing is the
character tokenizer shipped next to the weights (extra.py), so no GPL
phonemizer (espeak-ng) is involved anywhere, unlike Piper.

The TorchScript graph takes token ids only: no speaking-rate, noise or pitch
input exists (verified from the exported signature). Rate is therefore
applied after synthesis with ffmpeg's atempo (WSOLA, pitch-preserving) and
the emotion hook is limited to rate and energy -- see CONTRACTS.md #7.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
import unicodedata

from app.capabilities import require_language
from app.config import get_settings
from app.providers.base import SynthesisRequest, SynthesisResult, TTSProvider
from app.providers.registry import ProviderNotInstalledError
from app.providers.tts.common import fits_window, normalize, prosody_for, trim_silence

logger = logging.getLogger(__name__)

SAMPLE_RATE = 22050
_PUNCTUATION = "!¡'(),-.:;¿? "


class SyspinVoice:
    def __init__(self, repo_id: str) -> None:
        import torch
        from huggingface_hub import snapshot_download

        folder = snapshot_download(repo_id)
        spec = importlib.util.spec_from_file_location(f"syspin_extra_{abs(hash(repo_id))}", os.path.join(folder, "extra.py"))
        if spec is None or spec.loader is None:
            raise ProviderNotInstalledError("SyspinTTSProvider", f"a loadable extra.py in {folder}")
        extra = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extra)
        letters = open(os.path.join(folder, "chars.txt"), encoding="utf-8").read().strip("\n")
        config = extra.VitsConfig(
            text_cleaner="multilingual_cleaners",
            characters=extra.CharactersConfig(
                characters_class=extra.VitsCharacters, pad="<PAD>", eos="<EOS>", bos="<BOS>",
                blank="<BLNK>", characters=letters, punctuations=_PUNCTUATION, phonemes=None,
            ),
        )
        self.tokenizer, _ = extra.TTSTokenizer.init_from_config(config)
        self.letters = set(letters) | set(_PUNCTUATION)
        weights = [f for f in os.listdir(folder) if f.endswith(".pt")]
        if len(weights) != 1:
            raise ProviderNotInstalledError("SyspinTTSProvider", f"one TorchScript .pt in {folder}, found {weights}")
        self.net = torch.jit.load(os.path.join(folder, weights[0]), map_location="cpu").eval()
        self.repo_id = repo_id

    def render(self, text: str):
        import numpy as np
        import torch

        normalized = unicodedata.normalize("NFC", text)
        dropped = sorted({c for c in normalized if c not in self.letters and unicodedata.category(c)[0] in "LNM"})
        if dropped:
            logger.warning("TTS (%s) cannot pronounce and will skip: %s", self.repo_id, "".join(dropped))
        ids = self.tokenizer.text_to_ids(normalized)
        if not any(c in self.letters and unicodedata.category(c)[0] in "LM" for c in normalized):
            raise ValueError(f"{self.repo_id} cannot pronounce any character of {text!r}")
        with torch.inference_mode():
            return self.net(torch.from_numpy(np.array(ids)).unsqueeze(0)).squeeze().cpu().numpy().astype("float32")


class SyspinTTSProvider(TTSProvider):
    def __init__(self) -> None:
        try:
            import librosa  # noqa: F401
            import soundfile  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:
            raise ProviderNotInstalledError("SyspinTTSProvider", "torch, librosa, soundfile") from e
        self._settings = get_settings()
        self._voices: dict[str, SyspinVoice] = {}
        self._converter = None
        if self._settings.tts_voice_clone:
            from app.providers.tts.voice_clone import get_voice_converter

            self._converter = get_voice_converter()
        self._self_check(self._settings.default_target_language)

    def _voice_for(self, lang_code: str, gender: str | None) -> SyspinVoice:
        from app.capabilities import tts_voices

        lang = require_language(lang_code)
        candidates = [v for v in tts_voices(lang) if v.engine == "syspin"]
        if not candidates:
            raise ValueError(f"no SYSPIN voice is available for {lang.name} ({lang.code})")
        voice = next((v for v in candidates if v.gender == gender), candidates[0])
        if voice.model not in self._voices:
            self._voices[voice.model] = SyspinVoice(voice.model)
            logger.info("TTS: loaded %s (%s, %s)", voice.model, voice.gender, voice.license)
        return self._voices[voice.model]

    def _self_check(self, lang_code: str) -> None:
        import numpy as np

        from app.providers.loading import ModelLoadError

        voice = self._voice_for(lang_code, None)
        letters = [c for c in voice.letters if unicodedata.category(c)[0] == "L"][:16]
        wav = voice.render(" ".join("".join(letters[i:i + 4]) for i in range(0, len(letters), 4)))
        rms_db = 20 * np.log10(float(np.sqrt(np.mean(wav ** 2))) + 1e-12)
        if not np.isfinite(wav).all() or len(wav) < SAMPLE_RATE * 0.2 or rms_db < -50:
            raise ModelLoadError(f"SYSPIN self-check failed for {lang_code}: {len(wav) / SAMPLE_RATE:.2f}s, RMS {rms_db:.1f} dBFS")
        logger.info("TTS self-check passed for %s (%s)", lang_code, voice.repo_id)

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        import soundfile as sf

        from app.pipeline import ffmpeg_utils

        text = request.text.strip()
        if not text:
            raise ValueError("TTS was asked to speak an empty string")
        prosody = prosody_for(request.emotion, self._settings.emotion_confidence_floor)
        voice = self._voice_for(request.target_lang, request.extra.get("voice_gender"))
        wav = trim_silence(voice.render(text), SAMPLE_RATE)

        # No time fitting here: mux_export (timeline.plan_timeline) is the one
        # place a clip is sped up to its window, within one MAX_TEMPO budget.
        # Fitting here too compounded the two (1.25 x 1.35 = 1.69x), which is
        # what wrecked short lines -- CONTRACTS.md #7. A line that won't fit
        # is rendered at neutral rate so the emotion's rate change doesn't
        # add a second stretch pass on top of the mux's.
        rate = prosody.rate
        if not fits_window(len(wav) / SAMPLE_RATE / rate, request.target_duration_ms):
            rate = 1.0
        if abs(rate - 1.0) > 0.01:
            wav = ffmpeg_utils.time_stretch(wav, SAMPLE_RATE, rate)

        sr = SAMPLE_RATE
        if request.voice_reference_path:
            if self._converter is None:
                raise ValueError("voice cloning was requested but TTS_VOICE_CLONE is off on this worker")
            wav, sr = self._converter.convert(wav, sr, request.voice_reference_path, voice_key=voice.repo_id)
            wav = trim_silence(wav, sr)

        wav = normalize(wav, prosody.gain_db)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, wav, sr, subtype="PCM_16")
        return SynthesisResult(local_audio_path=path, duration_ms=int(1000 * len(wav) / sr), sample_rate=sr)
