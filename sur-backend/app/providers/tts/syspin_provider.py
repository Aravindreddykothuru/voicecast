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

import hashlib
import importlib.util
import logging
import os
import tempfile
import unicodedata

from app.capabilities import require_language
from app.config import get_settings
from app.providers.base import SynthesisRequest, SynthesisResult, TTSProvider
from app.providers.registry import ProviderNotInstalledError
from app.providers.tts.common import fits_window, normalize, prosody_for, render_line, trim_silence

logger = logging.getLogger(__name__)

SAMPLE_RATE = 22050
_PUNCTUATION = "!¡'(),-.:;¿? "


_BULLET = chr(10) + "  - "


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verified(path: str, want: str, what: str) -> str:
    """`path`, after checking it is byte-for-byte the pinned file."""
    from app.providers.loading import ModelLoadError

    got = _sha256(path)
    if got != want:
        raise ModelLoadError(f"{what}: sha256 {got[:16]} does not match the pinned {want[:16]} "
                             f"(app/providers/tts/syspin_manifest.py). Refusing to load a changed release.")
    return path


def _pinned_extra_py(folder: str, repo_id: str, pin) -> str:
    """This voice's extra.py, or the byte-identical one borrowed from a sibling
    release when the voice was published without it (issue #6)."""
    from app.providers.tts.syspin_manifest import EXTRA_PY_DONORS, EXTRA_PY_SHA256, pin_for

    if pin.ships_extra_py:
        return _verified(os.path.join(folder, "extra.py"), EXTRA_PY_SHA256, f"{repo_id}/extra.py")
    from huggingface_hub import hf_hub_download

    tried = []
    for donor in EXTRA_PY_DONORS:
        if donor == repo_id:
            continue
        try:
            path = hf_hub_download(donor, "extra.py", revision=pin_for(donor).revision)
        except Exception as e:  # noqa: BLE001 -- not cached, or no network
            tried.append(f"{donor} ({type(e).__name__})")
            continue
        _verified(path, EXTRA_PY_SHA256, f"{donor}/extra.py (borrowed for {repo_id})")
        logger.warning("SUPPLY CHAIN: %s was published without extra.py; loading the byte-identical file "
                       "from %s@%s instead. Surfaced in /api/capabilities as tts_voice_warnings.",
                       repo_id, donor, pin_for(donor).revision[:12])
        return path
    raise ProviderNotInstalledError(
        "SyspinTTSProvider",
        f"extra.py for {repo_id}: absent from the release, and no pinned sibling to borrow it from "
        f"(tried {', '.join(tried) or 'none'})",
    )


class SyspinVoice:
    def __init__(self, repo_id: str) -> None:
        import torch
        from huggingface_hub import snapshot_download

        from app.providers.tts.syspin_manifest import pin_for

        from app.providers.loading import ModelLoadError

        try:
            pin = pin_for(repo_id)
        except KeyError as e:
            # Not a missing dependency -- an unpinned artifact. Say which.
            raise ModelLoadError(str(e.args[0])) from None
        wanted = list(pin.files) + (["extra.py"] if pin.ships_extra_py else [])
        folder = snapshot_download(repo_id, revision=pin.revision, allow_patterns=wanted)
        for name, digest in pin.files.items():
            _verified(os.path.join(folder, name), digest, f"{repo_id}@{pin.revision[:12]}/{name}")
        spec = importlib.util.spec_from_file_location(
            f"syspin_extra_{abs(hash(repo_id))}", _pinned_extra_py(folder, repo_id, pin))
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
        weights = [f for f in pin.files if f.endswith(".pt")]
        if len(weights) != 1:
            raise ProviderNotInstalledError("SyspinTTSProvider", f"one pinned TorchScript .pt for {repo_id}, found {weights}")
        self.net = torch.jit.load(os.path.join(folder, weights[0]), map_location="cpu").eval()
        self.repo_id = repo_id

    def render(self, text: str, draw: int = 0):
        import numpy as np
        import torch

        normalized = unicodedata.normalize("NFC", text)
        dropped = sorted({c for c in normalized if c not in self.letters and unicodedata.category(c)[0] in "LNM"})
        if dropped:
            logger.warning("TTS (%s) cannot pronounce and will skip: %s", self.repo_id, "".join(dropped))
        ids = self.tokenizer.text_to_ids(normalized)
        if not any(c in self.letters and unicodedata.category(c)[0] in "LM" for c in normalized):
            raise ValueError(f"{self.repo_id} cannot pronounce any character of {text!r}")
        # Deterministic per text and draw, as the MMS engine already is: the
        # exported graph samples durations and noise from the global RNG, so
        # the seed is the only handle the TorchScript signature leaves. Without
        # it, re-rendering an unchanged line returns a different clip of a
        # different length -- which is exactly what /regenerate does.
        torch.manual_seed(int(hashlib.sha256(f"{draw}:{normalized}".encode()).hexdigest()[:8], 16))
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
        self._self_check_every_voice()

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

    def _self_check_every_voice(self) -> None:
        """Load and speak with every voice this deployment could select.

        Checking only the default language's first voice left a hole a
        release fell straight through: SYSPIN published BengaliFemale without
        its extra.py, so a Bengali dub crashed at synthesize for any speaker
        whose estimated F0 chose the female voice, while the worker had
        started clean. Which voice a job needs is decided by the speaker, not
        by configuration, so every voice the policy offers is loaded here --
        the cost is paid once at startup instead of once per customer."""
        from app.capabilities import SUPPORTED_LANGUAGES, tts_voices

        checked, failures = [], []
        for lang in SUPPORTED_LANGUAGES:
            for voice in tts_voices(lang):
                if voice.engine != "syspin":
                    continue
                try:
                    self._self_check(lang.code, voice.model)
                    checked.append(voice.model)
                except Exception as e:  # noqa: BLE001 -- report them all, not the first
                    failures.append(f"{lang.code}/{voice.gender or '?'} ({voice.model}): {e}")
        if failures:
            from app.providers.loading import ModelLoadError

            raise ModelLoadError(
                "SYSPIN voice(s) this deployment offers cannot speak:" + _BULLET
                + _BULLET.join(failures))
        logger.info("TTS self-check passed for %d voice(s), each verified against its pinned revision and "
                    "sha256: %s", len(checked), ", ".join(checked))
        from app.providers.tts.syspin_manifest import supply_chain_warnings

        for warning in supply_chain_warnings(checked):
            logger.warning("SUPPLY CHAIN: %s", warning)

    def _self_check(self, lang_code: str, model: str | None = None) -> None:
        import numpy as np

        from app.providers.loading import ModelLoadError

        if model is None:
            voice = self._voice_for(lang_code, None)
        else:
            if model not in self._voices:
                self._voices[model] = SyspinVoice(model)
            voice = self._voices[model]
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
        lang = require_language(request.target_lang)
        wav = render_line(voice.render, text, SAMPLE_RATE, carrier=lang.tts_carrier,
                          merge_closures=lang.tts_carrier_merge_closures)

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
