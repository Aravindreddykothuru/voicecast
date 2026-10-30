"""IndicF5 (ai4bharat/IndicF5, MIT): reference-clip voice cloning, in its own
interpreter (.venv-indicf5: needs transformers<4.50 and numpy<=1.26.4).

Follows the model card: model(text, ref_audio_path=..., ref_text=...) at
24 kHz. References come only from app/tts_runtime/data/tts_references.yaml,
and only entries whose licence and consent are recorded as `verified` are
used ("By using this model, you agree to only clone voices for which you
have explicit permission" -- model card).

Checked against the real model.py at the pinned revision
(docs/indicf5-adapter-diff.md). model.py imports the `f5_tts` package and
calls load_vocoder(..., is_local=False), which reaches for
charactr/vocos-mel-24khz over the network; the adapter serves that and the
repo's own checkpoints/vocab.txt from the verified local copies so
rendering stays offline.

.venv-indicf5 needs f5-tts, transformers<4.50 (config.json pins 4.49.0) and
numpy<=1.26.4. The f5-tts PACKAGE is MIT; the SWivid/F5-TTS WEIGHTS are
CC-BY-NC and are refused by name in licenses.FORBIDDEN. IndicF5 ships its
own MIT weights, which is what this loads.
"""
from __future__ import annotations

import os
from pathlib import Path

from app.tts_runtime.adapters.base import Adapter, SynthOutput, Unavailable, runtime_home

REFERENCES = Path(__file__).resolve().parents[1] / "data" / "tts_references.yaml"


def references(path: Path = REFERENCES) -> list[dict]:
    import yaml

    if not path.exists():
        return []
    return list((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("references", []))


def usable_references() -> list[dict]:
    return [r for r in references() if r.get("status") == "verified" and r.get("text") and r.get("file")]


class IndicF5Adapter(Adapter):
    name = "indicf5"
    repo = "ai4bharat/IndicF5"
    license = "MIT"
    languages = frozenset({"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"})
    supports_cpu_fallback = True
    needs_token_for_setup = True
    PINS = ("indicf5", "vocos")
    SR = 24000

    @classmethod
    def _pins(cls):
        from app.tts_runtime.manifest import load_pin

        return [load_pin(p) for p in cls.PINS]

    @classmethod
    def availability(cls, options: dict) -> tuple[bool, str]:
        from app.tts_runtime.manifest import provisioned

        home = Path(options.get("home") or runtime_home())
        for pin in cls._pins():
            ok, why = provisioned(home, pin)
            if not ok:
                hint = " (gated: needs HF_TOKEN, then make setup-tts)" if pin.gated else " (run make setup-tts)"
                return False, f"{pin.repo}: {why}{hint}"
        if not usable_references():
            return False, ("no reference clip with verified licence and consent in "
                           "app/tts_runtime/data/tts_references.yaml")
        return True, ""

    @classmethod
    def pick_reference(cls, lang: str, speaker: str | None, emotion: str | None) -> dict:
        refs = usable_references()
        if not refs:
            raise Unavailable("no verified reference clip")

        def score(r):
            return ((r.get("id") == speaker) * 8 + (r.get("lang") == lang) * 4 +
                    (r.get("gender") == speaker) * 2 + (r.get("emotion") == (emotion or "neutral")))

        return max(refs, key=score)

    @classmethod
    def version_for(cls, lang, speaker, options) -> str:
        try:
            ref = cls.pick_reference(lang, speaker, None)["id"]
        except Unavailable:
            ref = "none"
        return f"{cls._pins()[0].revision[:12]}:{ref}"

    def load(self, device: str) -> dict:
        import torch

        from app.tts_runtime.manifest import model_dir, verify_locked

        home = Path(self.options.get("home") or runtime_home())
        pins = self._pins()
        for pin in pins:
            verify_locked(model_dir(home, pin), pin)
        self.folder = model_dir(home, pins[0])
        vocos_dir = model_dir(home, pins[1])
        self._serve_locally(vocos_dir, self.folder)
        from transformers import AutoModel

        self.device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        # remove_sil and speed come from options, not the shipped config.json:
        # it defaults remove_sil=True, which keeps keep_silence=500 ms at each
        # edge and would fail the runtime's calibrated 0.30 s silence trigger
        # on every line (docs/indicf5-adapter-diff.md).
        overrides = {k: self.options[k] for k in ("remove_sil", "speed") if k in self.options}
        self.model = AutoModel.from_pretrained(str(self.folder), trust_remote_code=True, **overrides)
        self._assert_weights_loaded()
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        return {"device": self.device, "remove_sil": getattr(self.model.config, "remove_sil", None),
                "speed": getattr(self.model.config, "speed", None)}

    @staticmethod
    def _serve_locally(vocos_dir: Path, model_dir: Path) -> None:
        """Serve every hub fetch model.py makes from the verified local copies.

        Two of them. load_vocoder(is_local=False) pulls
        charactr/vocos-mel-24khz, and __init__ does

            hf_hub_download(config.name_or_path, "checkpoints/vocab.txt")

        which, loaded from a folder, asks the hub for a repo whose id is a
        local path -- and under HF_HUB_OFFLINE=1 simply fails. Both files are
        already provisioned and hash-locked, so both are answered from disk.
        """
        import huggingface_hub

        real = huggingface_hub.hf_hub_download

        def local_first(repo_id=None, filename=None, *a, **k):
            if filename:
                if repo_id == "charactr/vocos-mel-24khz" and (vocos_dir / filename).exists():
                    return str(vocos_dir / filename)
                # repo_id is "ai4bharat/IndicF5" or the local folder path.
                if (model_dir / filename).exists():
                    return str(model_dir / filename)
            return real(repo_id, filename, *a, **k)

        huggingface_hub.hf_hub_download = local_first
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    def _assert_weights_loaded(self) -> None:
        """Refuse to speak unless the checkpoint's tensors actually landed.

        CONTRACTS.md #1. Every key in model.safetensors carries a
        `_orig_mod.` segment, because __init__ wraps both submodules in
        torch.compile and the checkpoint was saved from the compiled model.
        torch.compile needs Triton, which Windows does not have. If the wrap
        degrades to a no-op the module tree has no `_orig_mod` level, not one
        of the 447 tensors matches, and from_pretrained *warns* and hands
        back a model speaking with untrained weights.
        """
        from safetensors import safe_open

        ckpt = self.folder / "model.safetensors"
        with safe_open(str(ckpt), framework="pt") as f:
            expected = set(f.keys())
        got = set(self.model.state_dict().keys())
        missing = expected - got
        if missing:
            raise Unavailable(
                f"IndicF5 weights did not load: {len(missing)} of {len(expected)} tensors from "
                f"model.safetensors are absent from the built model (e.g. {sorted(missing)[0]}). "
                f"The checkpoint's keys carry a '_orig_mod.' segment from torch.compile; if compile "
                f"is a no-op here the names cannot match and the model would speak untrained. "
                f"Refusing rather than rendering.")

    def synth(self, text, lang, speaker, emotion, draw=0) -> SynthOutput:
        import hashlib

        import numpy as np
        import torch

        ref = self.pick_reference(lang, speaker, emotion)
        # sha256, not hash(): Python randomises hash() per process, and the
        # same line must render the same way after a restart.
        torch.manual_seed(int(hashlib.sha256(f"{draw}:{ref['id']}:{text}".encode()).hexdigest()[:8], 16))
        audio = self.model(text, ref_audio_path=str(self.folder / ref["file"]), ref_text=ref["text"])
        audio = np.asarray(audio)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        return SynthOutput(audio.astype(np.float32), self.SR,
                           {"reference": ref["id"], "emotion_applied": ref.get("emotion") == emotion})
