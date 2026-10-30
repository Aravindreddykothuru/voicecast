"""SyspinTTSProvider, except the raw SYSPIN render runs in an isolated
subprocess (app.tts_runtime.pool.ModelPool + adapters.syspin.SyspinAdapter)
instead of in-process.

Why: a crash, hang, or a runaway TorchScript call inside `voice.render()`
used to take the whole Celery worker process down with it. The runtime
already isolates this in its own worker process, with a timeout and an OOM
ladder, and has been proven (tests/test_tts_runtime_real.py) to render
byte-identical audio to `common.render_line` for draw=0 -- the only draw
production ever asks for. This class swaps only that one step; every other
part of `synthesize()` (prosody rate, time-fit, voice cloning, loudness
normalize) is untouched, inherited from `SyspinTTSProvider` unchanged.

Feature-flagged: `TTS_USE_RUNTIME=true` (app/config.py). One-line rollback:
set it back to false (the default) and restart the worker -- no code change.

This is wiring, not a runtime change: nothing in app/tts_runtime/ is
imported here except its public, already-tested entry points (RuntimeConfig,
ModelPool, runtime_home). See CONTRACTS.md #9 and #7, and
docs/tts-runtime.md's "Wired into the pipeline" section.
"""
from __future__ import annotations

import logging
import os
import tempfile

from app.providers.tts.syspin_provider import SAMPLE_RATE, SyspinTTSProvider, VoiceHandle

logger = logging.getLogger(__name__)


class _VoiceRef:
    """All `synthesize()` still needs from a voice once rendering moved to the
    worker: the repo id, used as the voice-cloning converter's cache key."""

    __slots__ = ("repo_id",)

    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id


class RuntimeSyspinTTSProvider(SyspinTTSProvider):
    def __init__(self) -> None:
        super().__init__()
        from app.tts_runtime import config as rt_config
        from app.tts_runtime.adapters.base import runtime_home
        from app.tts_runtime.pool import ModelPool

        # __init__ above loaded and spoke every voice this deployment offers
        # (CONTRACTS.md #1 -- the BengaliFemale incident), which is still the
        # right startup check: bad weights must fail the worker, not a
        # customer's job. But nothing in this class renders in-process, so
        # holding those ~10 checkpoints here is dead weight on top of the
        # copies the subprocess loads for itself. Keeping both was enough to
        # get the worker killed mid-load while it loaded an 11th.
        self._voices.clear()
        self._rt_home = runtime_home()
        self._rt_cfg = rt_config.load()
        self._pool = ModelPool(self._rt_cfg, self._rt_home, self._rt_options_for)
        logger.info("TTS: SYSPIN renders will run in an isolated subprocess (TTS_USE_RUNTIME=true), "
                    "timeout %.0fs, home %s", self._rt_cfg.synth_timeout_for("syspin"), self._rt_home)

    def _voice_for(self, lang_code: str, gender: str | None) -> VoiceHandle:
        """Pick the voice without loading it: the worker loads and renders.
        Deliberately the adapter's own selector, so the repo this reports and
        the repo the worker actually renders with cannot drift apart."""
        from app.tts_runtime.adapters.syspin import _repo_for

        return _VoiceRef(_repo_for(lang_code, gender))

    def _rt_options_for(self, model_id: str) -> dict:
        # Mirrors Runner.options_for exactly, so the subprocess sees the same
        # options a batch job would give it (tts_chains.yaml's carrier map).
        return {"carrier": self._rt_cfg.uses_carrier(model_id), "home": str(self._rt_home)}

    def _render_raw(self, voice: VoiceHandle, text: str, lang, gender: str | None):
        import numpy as np
        import soundfile as sf

        fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="rt-syspin-")
        os.close(fd)
        try:
            # draw=0: the runtime's SyspinAdapter maps draw=0 to base offset 0,
            # so this calls the identical render_line(voice.render, ...) the
            # in-process path calls -- proven byte-identical (draw=0 only).
            self._pool.synth("syspin", text, lang.code, gender, None, draw=0, out=out_path)
            wav, sr = sf.read(out_path, dtype="float32")
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"runtime SYSPIN worker returned {sr} Hz, expected {SAMPLE_RATE}")
        return np.ascontiguousarray(wav)

    def close(self) -> None:
        """Shut down the subprocess. Not called in production (the provider
        lives for the worker's lifetime); here for tests and clean shutdown."""
        self._pool.close_all()
