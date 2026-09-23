"""A test double that fails on command. Loads no weights.

options["behavior"]:
  default: mode for every line (default "ok")
  texts:   {text: mode} overrides
  sleep_s: delay before answering
  state_dir: where once-only modes remember they fired (survives restarts)
  languages: languages it claims (default: all)
  available: false -> reports itself unavailable
  pins / pins_dir: pinned "model" files verified at load (corrupted-file chaos test)

Modes: ok, crash (process exits), hang, garbage (10 s of noise), nan, empty,
exception, oom (always), oom_once, oom_on_gpu (OOM unless loaded on cpu),
clip (full-scale square wave), lead_silence (2.5 s of silence first),
crash_once (exits the first time only), network (opens a connection to
huggingface.co -- proves the offline guard is on when it fails).
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

from app.tts_runtime.adapters.base import Adapter, SynthOutput

SR = 16000
SECONDS_PER_CHAR = 0.08


def speechlike(text: str, draw: int = 0) -> np.ndarray:
    """A voiced, amplitude-modulated harmonic signal ~0.08 s per character."""
    n_chars = max(1, sum(1 for c in text if c.isalnum() or 0x0900 <= ord(c) <= 0x0DFF))
    dur = max(0.25, SECONDS_PER_CHAR * n_chars)
    t = np.arange(int(dur * SR)) / SR
    f0 = 140 + 10 * (int(hashlib.sha256(f"{draw}:{text}".encode()).hexdigest()[:4], 16) % 5)
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    env = 0.5 * (1 - np.cos(2 * np.pi * np.minimum(t / dur, 1.0))) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    pad = np.zeros(int(0.05 * SR))
    return np.concatenate([pad, 0.3 * sig * env, pad]).astype(np.float32)


class FakeAdapter(Adapter):
    name = "fake"
    repo = "local/fake"
    license = "LicenseRef-no-weights"
    languages = frozenset({"hi", "mr", "bn", "te", "kn", "ta", "ml", "gu", "pa", "or", "as", "ur", "en"})
    supports_cpu_fallback = True

    @classmethod
    def _behavior(cls, options: dict) -> dict:
        return (options or {}).get("behavior", {}) or {}

    @classmethod
    def supports(cls, lang: str) -> bool:
        return lang in cls.languages

    @classmethod
    def _pins(cls, options: dict):
        from app.tts_runtime.provision import _load, pins_for

        names, pins_dir = pins_for("fake", options)
        return [_load(n, pins_dir) for n in names]

    @classmethod
    def availability(cls, options: dict) -> tuple[bool, str]:
        b = cls._behavior(options)
        if b.get("available", True) is False:
            return False, "fake configured unavailable"
        if b.get("pins"):
            from app.tts_runtime.manifest import provisioned

            home = Path(options.get("home") or ".")
            for pin in cls._pins(options):
                ok, why = provisioned(home, pin)
                if not ok:
                    return False, f"{pin.repo}: {why}"
        return True, ""

    @classmethod
    def version_for(cls, lang, speaker, options) -> str:
        return "fake-1:" + str(sorted(cls._behavior(options).items()))[:64]

    def load(self, device: str) -> dict:
        self.device = device
        b = self._behavior(self.options)
        if b.get("pins"):
            from app.tts_runtime.manifest import model_dir, verify_locked

            home = Path(self.options.get("home") or ".")
            for pin in self._pins(self.options):
                verify_locked(model_dir(home, pin), pin)      # ModelCorrupt -> never loaded
        if b.get("load") == "crash":
            os._exit(4)
        if b.get("load") == "oom_on_gpu" and device != "cpu":
            raise RuntimeError("CUDA out of memory while loading (fake)")
        if b.get("load") == "exception":
            raise RuntimeError("fake: load failed on purpose")
        return {"device": device, "variant": self.variant}

    def _once(self, tag: str) -> bool:
        """True the first time `tag` is seen (per state_dir), False after."""
        d = Path(self._behavior(self.options).get("state_dir") or ".")
        d.mkdir(parents=True, exist_ok=True)
        marker = d / f"fired-{self.variant}-{tag}"
        if marker.exists():
            return False
        marker.write_text("1")
        return True

    def synth(self, text, lang, speaker, emotion, draw=0) -> SynthOutput:
        b = self._behavior(self.options)
        if b.get("sleep_s"):
            time.sleep(float(b["sleep_s"]))
        mode = (b.get("texts") or {}).get(text, b.get("default", "ok"))
        if mode == "crash" or (mode == "crash_once" and self._once("crash:" + text)):
            os._exit(3)
        if mode == "hang":
            time.sleep(10_000)
        if mode == "exception":
            raise ValueError(f"fake: exception on purpose for {text!r}")
        if mode == "network":
            # Tries to reach the internet; succeeds only if nothing blocks it.
            import socket

            socket.create_connection(("huggingface.co", 443), timeout=5).close()
        if mode == "oom" or (mode == "oom_once" and self._once("oom:" + text)) or \
                (mode == "oom_on_gpu" and self.device != "cpu"):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB (fake)")
        if mode == "nan":
            return SynthOutput(np.full(SR, np.nan, dtype=np.float32), SR)
        if mode == "empty":
            return SynthOutput(np.zeros(0, dtype=np.float32), SR)
        if mode == "garbage":
            rng = np.random.default_rng(0)
            return SynthOutput((0.3 * rng.standard_normal(10 * SR)).astype(np.float32), SR)
        if mode == "clip":
            t = np.arange(int(0.08 * max(3, len(text)) * SR)) / SR
            return SynthOutput(np.sign(np.sin(2 * np.pi * 150 * t)).astype(np.float32), SR)
        x = speechlike(text, draw)
        if mode == "lead_silence":
            x = np.concatenate([np.zeros(int(2.5 * SR), dtype=np.float32), x])
        return SynthOutput(x, SR, {"device": self.device, "variant": self.variant})

    def health_check(self) -> dict:
        b = self._behavior(self.options)
        if b.get("health") == "fail":
            return {"ok": False, "detail": "fake: health check fails on purpose"}
        return {"ok": True}
