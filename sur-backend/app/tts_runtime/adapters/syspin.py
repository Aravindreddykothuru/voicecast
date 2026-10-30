"""SYSPIN VITS (CC-BY-4.0), pinned and hash-checked (syspin_manifest.py).

Renders exactly what the pipeline renders today: common.render_line with the
language's carrier and closure-merge flag, median of three deterministic
draws for short lines. A retry asks for a different set of draws.
"""
from __future__ import annotations

import math

import numpy as np

from app.tts_runtime.adapters.base import Adapter, SynthOutput

# Bump when the rendering code path changes the audio for the same text, so
# the render cache (keyed on version) cannot serve stale clips.
RENDER_REV = "r2-closure-merge"
DRAW_STRIDE = 10  # retry k uses draws 10k, 10k+1, 10k+2


def _voices(lang: str) -> dict[str, str]:
    from app.capabilities import SUPPORTED_LANGUAGES

    for lang_obj in SUPPORTED_LANGUAGES:
        if lang_obj.code == lang:
            return {v.gender: v.model for v in lang_obj.voices if v.engine == "syspin"}
    return {}


def _repo_for(lang: str, speaker: str | None) -> str:
    voices = _voices(lang)
    if not voices:
        raise ValueError(f"no SYSPIN voice for {lang}")
    if speaker in voices:
        return voices[speaker]
    return voices.get("male") or next(iter(voices.values()))


class SyspinAdapter(Adapter):
    name = "syspin"
    repo = "SYSPIN/tts_vits_coquiai_HindiFemale"   # every voice matches SYSPIN/tts_vits_coquiai_*
    license = "CC-BY-4.0"
    languages = frozenset({"hi", "mr", "bn", "te", "kn"})

    @classmethod
    def availability(cls, options: dict) -> tuple[bool, str]:
        try:
            from huggingface_hub import try_to_load_from_cache

            from app.providers.tts.syspin_manifest import SYSPIN_MANIFEST
            from app.tts_runtime import licenses
        except ImportError as e:
            return False, f"dependencies missing: {e}"
        missing = []
        for repo, pin in SYSPIN_MANIFEST.items():
            licenses.check("syspin", repo, cls.license)
            for f in pin.files:
                hit = try_to_load_from_cache(repo, f, revision=pin.revision)
                if not isinstance(hit, str):
                    missing.append(f"{repo}/{f}")
        if missing:
            return False, f"not provisioned: {len(missing)} pinned file(s) not cached, e.g. {missing[0]}"
        return True, ""

    @classmethod
    def version_for(cls, lang, speaker, options) -> str:
        from app.capabilities import require_language
        from app.providers.tts.syspin_manifest import pin_for

        repo = _repo_for(lang, speaker)
        lg = require_language(lang)
        carrier = bool(options.get("carrier"))
        merge = carrier and lg.tts_carrier_merge_closures
        return f"{repo.rsplit('_', 1)[-1]}@{pin_for(repo).revision[:12]}:{RENDER_REV}:carrier={int(carrier)}:merge={int(merge)}"

    @classmethod
    def sha256_manifest(cls, options: dict) -> dict[str, str]:
        from app.providers.tts.syspin_manifest import SYSPIN_MANIFEST

        return {f"{repo}@{pin.revision}/{f}": h for repo, pin in SYSPIN_MANIFEST.items() for f, h in pin.files.items()}

    def load(self, device: str) -> dict:
        import torch  # noqa: F401 -- fail here, not mid-line, if torch is missing

        self.device = "cpu"          # TorchScript export, CPU only
        self._voices: dict[str, object] = {}
        return {"device": self.device}

    def _voice(self, repo: str):
        if repo not in self._voices:
            from app.providers.tts.syspin_provider import SyspinVoice

            self._voices[repo] = SyspinVoice(repo)
        return self._voices[repo]

    def synth(self, text, lang, speaker, emotion, draw=0) -> SynthOutput:
        from app.capabilities import require_language
        from app.providers.tts.common import render_line
        from app.providers.tts.syspin_provider import SAMPLE_RATE

        repo = _repo_for(lang, speaker)
        voice = self._voice(repo)
        base = DRAW_STRIDE * int(draw)

        def render(t, draw=0):
            return voice.render(t, draw=base + draw)

        lg = require_language(lang)
        carrier = lg.tts_carrier if self.options.get("carrier") else None
        wav = render_line(render, text, SAMPLE_RATE, carrier=carrier,
                          merge_closures=bool(carrier) and lg.tts_carrier_merge_closures)
        # emotion: SYSPIN has no emotion control; the caller records that.
        return SynthOutput(np.asarray(wav, dtype=np.float32), SAMPLE_RATE,
                           {"voice": repo, "emotion_applied": False})

    def health_check(self) -> dict:
        try:
            out = self.synth("नमस्ते", "hi", None, None)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "detail": f"{type(e).__name__}: {e}"}
        ok = len(out.samples) > 0.2 * out.sr and bool(np.isfinite(out.samples).all()) and \
            20 * math.log10(float(np.sqrt(np.mean(out.samples ** 2))) + 1e-9) > -50
        return {"ok": ok, "detail": f"{len(out.samples) / out.sr:.2f}s"}
