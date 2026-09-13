"""Provider factory: reads *_PROVIDER env vars and returns the right
implementation. This is the ONLY place that should branch on "mock" vs
"real" -- everything else (pipeline tasks, tests) just calls
`get_asr_provider()` etc. and gets back something satisfying the ABC.

Real implementations import their heavy dependencies lazily, inside
`__init__`, so `docker-compose up` with every *_PROVIDER=mock never even
tries to import torch/transformers/etc.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.providers.base import ASRProvider, DiarizationProvider, EmotionProvider, TranslationProvider, TTSProvider


class ProviderNotInstalledError(RuntimeError):
    def __init__(self, provider_name: str, package_hint: str) -> None:
        super().__init__(
            f"{provider_name} is configured (PROVIDER=real) but its dependencies aren't "
            f"installed. Run `pip install -r requirements-ml.txt` ({package_hint}) on a "
            f"GPU-capable worker, or set the corresponding *_PROVIDER back to 'mock'."
        )


@lru_cache
def get_diarization_provider() -> DiarizationProvider:
    settings = get_settings()
    if settings.diarization_provider == "mock":
        from app.providers.diarization.mock_provider import MockDiarizationProvider

        return MockDiarizationProvider()
    from app.providers.diarization.pyannote_provider import PyannoteDiarizationProvider

    return PyannoteDiarizationProvider()


@lru_cache
def get_asr_provider() -> ASRProvider:
    settings = get_settings()
    if settings.asr_provider == "mock":
        from app.providers.asr.mock_provider import MockASRProvider

        return MockASRProvider()
    from app.providers.asr.faster_whisper_provider import FasterWhisperASRProvider

    return FasterWhisperASRProvider()


@lru_cache
def get_emotion_provider() -> EmotionProvider:
    settings = get_settings()
    if settings.emotion_provider == "mock":
        from app.providers.emotion.mock_provider import MockEmotionProvider

        return MockEmotionProvider()
    from app.providers.emotion.wav2vec2_provider import Wav2Vec2EmotionProvider

    return Wav2Vec2EmotionProvider()


@lru_cache
def get_translation_provider() -> TranslationProvider:
    settings = get_settings()
    if settings.translation_provider == "mock":
        from app.providers.translation.mock_provider import MockTranslationProvider

        return MockTranslationProvider()
    from app.providers.translation.indictrans2_provider import IndicTrans2Provider

    return IndicTrans2Provider()


@lru_cache
def get_tts_provider() -> TTSProvider:
    settings = get_settings()
    if settings.tts_provider == "mock":
        from app.providers.tts.mock_provider import MockTTSProvider

        return MockTTSProvider()
    # TTS_ENGINE picks the renderer; voice cloning (if enabled) is layered on
    # by the provider. Licenses per engine: app/capabilities.py, CONTRACTS.md #7.
    if settings.tts_engine == "syspin":
        from app.providers.tts.syspin_provider import SyspinTTSProvider

        return SyspinTTSProvider()
    if settings.tts_require_commercial_license:
        raise RuntimeError(
            f"TTS_ENGINE={settings.tts_engine} uses non-commercially licensed weights, but "
            "TTS_REQUIRE_COMMERCIAL_LICENSE is true. Use TTS_ENGINE=syspin, or set "
            "TTS_REQUIRE_COMMERCIAL_LICENSE=false for research/non-commercial deployments only."
        )
    from app.providers.tts.mms_provider import MMSTTSProvider

    return MMSTTSProvider()
