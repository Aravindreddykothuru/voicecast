"""CosyVoice2 voice conversion: re-voice rendered speech as a target speaker.

Uses only CosyVoice2's speech tokenizer, flow-matching decoder and vocoder
(`inference_vc`) -- no text model -- so unlike CosyVoice2 text-to-speech it
does not depend on the language having been in its training text. Runs in
the TTS venv (.venv-tts / Dockerfile.worker-tts); see requirements-tts.txt
for why that venv is separate.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile

from app.config import get_settings, resolve_backend_path
from app.providers.registry import ProviderNotInstalledError

logger = logging.getLogger(__name__)

# CosyVoice2 asserts both the source and the reference are <= 30s.
_MAX_SOURCE_S = 28.0
_MAX_REFERENCE_S = 30.0


def ensure_cosyvoice_importable(src_dir: str | None) -> None:
    if not src_dir:
        return
    root = resolve_backend_path(src_dir)
    if not os.path.isdir(os.path.join(root, "cosyvoice")):
        raise ProviderNotInstalledError(
            "CosyVoice2", f"COSYVOICE_SRC_DIR={src_dir!r} has no cosyvoice/ package (resolved to {root})"
        )
    for path in (root, os.path.join(root, "third_party", "Matcha-TTS")):
        if path not in sys.path:
            sys.path.insert(0, path)


class CosyVoiceVoiceConverter:
    def __init__(self) -> None:
        settings = get_settings()
        ensure_cosyvoice_importable(settings.cosyvoice_src_dir)
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
        except ImportError as e:
            raise ProviderNotInstalledError(
                "CosyVoice2 voice conversion",
                "the CosyVoice source checkout (COSYVOICE_SRC_DIR) plus requirements-tts.txt",
            ) from e

        model_dir = resolve_backend_path(settings.tts_model_name)
        if not os.path.isfile(os.path.join(model_dir, "cosyvoice2.yaml")):
            # CosyVoice2() treats a missing directory as a ModelScope repo id
            # and tries to download it -- a slow, confusing failure on a
            # worker whose weights are supposed to be provisioned already.
            raise ProviderNotInstalledError(
                "CosyVoice2 voice conversion",
                f"weights at TTS_MODEL_NAME={settings.tts_model_name!r} (no cosyvoice2.yaml in {model_dir})",
            )
        self._engine = CosyVoice2(model_dir)
        self.sample_rate = self._engine.sample_rate

    def convert(self, wav, sr: int, reference_path: str, voice_key: str | None = None):
        import numpy as np
        import soundfile as sf
        import torch

        ref_seconds = sf.info(reference_path).duration
        if ref_seconds > _MAX_REFERENCE_S:
            raise ValueError(f"voice reference is {ref_seconds:.1f}s; CosyVoice2 accepts at most 30s")

        window = int(_MAX_SOURCE_S * sr)
        out = []
        with tempfile.TemporaryDirectory(prefix="sur-vc-") as tmp:
            for n, start in enumerate(range(0, len(wav), window)):
                piece = os.path.join(tmp, f"src_{n}.wav")
                sf.write(piece, wav[start:start + window], sr, subtype="PCM_16")
                chunks = list(self._engine.inference_vc(piece, reference_path))
                if not chunks:
                    raise RuntimeError("CosyVoice2 voice conversion produced no audio")
                out.append(torch.cat([c["tts_speech"] for c in chunks], dim=1).numpy()[0])
        return np.concatenate(out).astype("float32"), self.sample_rate
