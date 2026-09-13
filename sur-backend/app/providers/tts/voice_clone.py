"""Voice cloning: re-voice rendered TTS speech as the original speaker.

Two converters, selected by TTS_VOICE_CLONE_ENGINE:

  openvoice  OpenVoice V2 tone-colour converter (MIT). A VITS-style converter
             that swaps the speaker embedding and keeps the content; fast on CPU.
  cosyvoice  CosyVoice2 voice conversion (Apache-2.0). Kept for comparison;
             measured slower and less intelligible on Indic output.

Whichever is selected must pass a budget enforced in code at worker startup
(check_budget): conversion faster than TTS_VOICE_CLONE_MAX_RTF times real
time, and content preserved (the converted clip's energy envelope tracks the
source and its duration is unchanged). A converter that fails refuses to
start, so clone_voice is never offered on hardware or weights that can't
deliver it. The numbers behind the choice are in CONTRACTS.md #7.
"""
from __future__ import annotations

import logging
import os
import sys
import time

from app.config import get_settings, resolve_backend_path
from app.providers.registry import ProviderNotInstalledError

logger = logging.getLogger(__name__)


class VoiceCloneBudgetError(RuntimeError):
    """The configured voice converter is too slow or destroys content."""


def get_voice_converter():
    engine = get_settings().tts_voice_clone_engine
    if engine == "openvoice":
        converter = OpenVoiceConverter()
    else:
        from app.providers.tts.cosyvoice_vc import CosyVoiceVoiceConverter

        converter = CosyVoiceVoiceConverter()
    check_budget(converter)
    return converter


def check_budget(converter, *, clock=time.perf_counter) -> None:
    """Convert a fixed synthetic utterance and enforce latency and content
    preservation before the worker accepts any cloning job.

    Timed with perf_counter after one untimed warm-up call: the first call
    pays one-off thread-pool and allocator start-up that no real job sees,
    and wall-clock time jumps across a laptop's sleep/resume."""
    import numpy as np

    settings = get_settings()
    sr = 22050
    t = np.arange(int(sr * 3.0)) / sr
    # Syllable-rate amplitude modulation over a harmonic source: speech-like
    # enough to exercise the converter, identical on every machine.
    f0 = 140 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    carrier = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = np.clip(np.sin(2 * np.pi * 4.0 * t), 0, None) ** 1.5
    source = (0.3 * carrier * envelope).astype("float32")
    reference = (0.3 * sum(np.sin(k * 2 * np.pi * 210 * t) / k for k in range(1, 8)) * (0.6 + 0.4 * envelope)).astype("float32")

    import tempfile

    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="sur-clone-check-") as tmp:
        ref_path = os.path.join(tmp, "ref.wav")
        sf.write(ref_path, reference, sr)
        converter.convert(source[: sr // 2], sr, ref_path, voice_key="__budget_warmup__")
        started = clock()
        out, out_sr = converter.convert(source, sr, ref_path, voice_key="__budget_check__")
        elapsed = clock() - started

    rtf = elapsed / 3.0
    duration_ratio = (len(out) / out_sr) / 3.0
    corr = _envelope_correlation(source, sr, out, out_sr)
    logger.info("voice clone budget: RTF %.2f (max %.2f), duration x%.3f, envelope corr %.2f",
                rtf, settings.tts_voice_clone_max_rtf, duration_ratio, corr)
    problems = []
    if rtf > settings.tts_voice_clone_max_rtf:
        problems.append(f"converts at {rtf:.2f}x real time, budget is {settings.tts_voice_clone_max_rtf:.2f}x")
    if not 0.9 <= duration_ratio <= 1.1:
        problems.append(f"changes duration by x{duration_ratio:.2f}")
    if corr < 0.6:
        problems.append(f"output energy envelope does not follow the source (corr {corr:.2f})")
    if problems:
        raise VoiceCloneBudgetError(
            f"voice cloning ({get_settings().tts_voice_clone_engine}) is outside its budget: "
            + "; ".join(problems) + ". Disable TTS_VOICE_CLONE or use a faster converter."
        )


def _envelope_correlation(a, sr_a: int, b, sr_b: int) -> float:
    import librosa
    import numpy as np

    hop = 0.02
    ea = librosa.feature.rms(y=a, frame_length=int(sr_a * 0.04), hop_length=int(sr_a * hop))[0]
    eb = librosa.feature.rms(y=b, frame_length=int(sr_b * 0.04), hop_length=int(sr_b * hop))[0]
    n = min(len(ea), len(eb))
    if n < 10:
        return 0.0
    return float(np.corrcoef(ea[:n], eb[:n])[0, 1])


class OpenVoiceConverter:
    CHECKPOINT_REPO = "myshell-ai/OpenVoiceV2"

    def __init__(self) -> None:
        settings = get_settings()
        if settings.openvoice_src_dir:
            root = resolve_backend_path(settings.openvoice_src_dir)
            if not os.path.isdir(os.path.join(root, "openvoice")):
                raise ProviderNotInstalledError("OpenVoiceConverter", f"OPENVOICE_SRC_DIR={settings.openvoice_src_dir!r} has no openvoice/ package")
            if root not in sys.path:
                sys.path.insert(0, root)
        try:
            import torch
            from huggingface_hub import snapshot_download
            from openvoice import utils  # type: ignore
            from openvoice.mel_processing import spectrogram_torch  # type: ignore
            from openvoice.models import SynthesizerTrn  # type: ignore
        except ImportError as e:
            raise ProviderNotInstalledError("OpenVoiceConverter", "the OpenVoice source checkout (OPENVOICE_SRC_DIR) plus torch/librosa") from e

        from app.providers.loading import ModelLoadError

        folder = os.path.join(snapshot_download(self.CHECKPOINT_REPO, allow_patterns=["converter/*"]), "converter")
        hps = utils.get_hparams_from_file(os.path.join(folder, "config.json"))
        model = SynthesizerTrn(len(getattr(hps, "symbols", [])), hps.data.filter_length // 2 + 1,
                               n_speakers=hps.data.n_speakers, **hps.model).eval()
        state = torch.load(os.path.join(folder, "checkpoint.pth"), map_location="cpu")
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        # The converter checkpoint omits the text encoder (enc_p), which
        # voice conversion never calls. Anything else missing means a broken
        # download or an incompatible release -- refuse (CONTRACTS.md #1).
        fatal = [k for k in missing if not k.startswith("enc_p.")]
        if fatal or unexpected:
            raise ModelLoadError(f"OpenVoice V2 converter weights incomplete: missing={fatal[:5]} unexpected={unexpected[:5]}")
        self._torch = torch
        self._spectrogram = spectrogram_torch
        self._hps = hps
        self._model = model
        self.sample_rate = hps.data.sampling_rate
        self._reference_se: dict[str, object] = {}
        self._voice_se: dict[str, tuple[object, int]] = {}

    def _spec(self, audio):
        y = self._torch.FloatTensor(audio).unsqueeze(0)
        h = self._hps.data
        return self._spectrogram(y, h.filter_length, self.sample_rate, h.hop_length, h.win_length, center=False)

    def _embed(self, audio):
        with self._torch.no_grad():
            return self._model.ref_enc(self._spec(audio).transpose(1, 2)).unsqueeze(-1)

    def convert(self, wav, sr: int, reference_path: str, voice_key: str | None = None):
        import librosa
        import numpy as np

        audio = librosa.resample(np.asarray(wav, dtype="float32"), orig_sr=sr, target_sr=self.sample_rate) if sr != self.sample_rate else np.asarray(wav, dtype="float32")
        key = f"{reference_path}:{os.path.getmtime(reference_path)}"
        if key not in self._reference_se:
            ref, _ = librosa.load(reference_path, sr=self.sample_rate)
            self._reference_se[key] = self._embed(ref)
        tgt_se = self._reference_se[key]

        # The source voice's embedding is a running mean over every clip that
        # voice has rendered: one short line alone gives a noisy estimate,
        # and the converter subtracts whatever it thinks the source timbre is.
        clip_se = self._embed(audio)
        total, n = self._voice_se.get(voice_key or "", (clip_se * 0, 0))
        total, n = total + clip_se, n + 1
        self._voice_se[voice_key or ""] = (total, n)
        src_se = total / n

        with self._torch.no_grad():
            spec = self._spec(audio)
            out = self._model.voice_conversion(spec, self._torch.LongTensor([spec.size(-1)]),
                                               sid_src=src_se, sid_tgt=tgt_se, tau=0.3)[0][0, 0]
        return out.cpu().numpy().astype("float32"), self.sample_rate
