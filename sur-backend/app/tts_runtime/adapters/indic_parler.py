"""Indic Parler-TTS (ai4bharat/indic-parler-tts, Apache-2.0): named speakers
and emotion prompts, in its own interpreter (.venv-parler: parler-tts pins
transformers==4.46.1).

Usage follows the model card exactly: the prompt is tokenised with the
model's tokenizer, the voice description with the text encoder's
(google/flan-t5-large), both loaded from the verified local copies.
Speakers and emotion support are the model card's; emotions are prompted
only in the ten languages the card says support them.

The weights are gated: nothing here has run against them on this machine
(no HF_TOKEN). The adapter's code path is exercised end to end by
ParlerTinyAdapter on the ungated parler-tts-tiny-v1 (English only).
"""
from __future__ import annotations

import hashlib

from app.tts_runtime.adapters.base import Adapter, SynthOutput, runtime_home

# Model card, "Recommended Speakers" (huggingface.co/ai4bharat/indic-parler-tts).
SPEAKERS: dict[str, tuple[str, ...]] = {
    "as": ("Amit", "Sita"), "bn": ("Arjun", "Aditi"), "gu": ("Yash", "Neha"), "hi": ("Rohit", "Divya"),
    "kn": ("Suresh", "Anu"), "ml": ("Anjali", "Harish"), "mr": ("Sanjay", "Sunita"), "or": ("Manas", "Debjani"),
    "pa": ("Divjot", "Gurpreet"), "ta": ("Jaya",), "te": ("Prakash", "Lalitha"), "en": ("Thoma", "Mary"),
}
# Model card: emotions are officially supported in these languages only.
EMOTION_LANGS = frozenset({"as", "bn", "brx", "doi", "kn", "ml", "mr", "sa", "ne", "ta"})
EMOTIONS = {"anger": "angry", "sadness": "sad", "happiness": "happy", "fear": "fearful",
            "surprise": "surprised", "neutral": None}
DESCRIPTION_REV = "d1"   # bump when the description template changes


def describe(name: str | None, emotion_word: str | None) -> str:
    who = f"{name}'s voice" if name else "A clear voice"
    tone = f"sounds {emotion_word}" if emotion_word else "is calm and neutral"
    return (f"{who} {tone}, with a moderate speed and pitch. The recording is of very high quality, "
            f"with the speaker's voice sounding clear and very close up.")


class IndicParlerAdapter(Adapter):
    name = "indic_parler"
    repo = "ai4bharat/indic-parler-tts"
    license = "Apache-2.0"
    languages = frozenset({"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"})
    supports_cpu_fallback = True
    needs_token_for_setup = True
    PINS = ("indic_parler", "flan_t5_large_tokenizer")
    SHARED_TOKENIZER = False

    @classmethod
    def _pins(cls):
        from app.tts_runtime.manifest import load_pin

        return [load_pin(p) for p in cls.PINS]

    @classmethod
    def availability(cls, options: dict) -> tuple[bool, str]:
        from pathlib import Path

        from app.tts_runtime.manifest import provisioned

        home = Path(options.get("home") or runtime_home())
        for pin in cls._pins():
            ok, why = provisioned(home, pin)
            if not ok:
                hint = " (gated: needs HF_TOKEN, then make setup-tts)" if pin.gated else " (run make setup-tts)"
                return False, f"{pin.repo}: {why}{hint}"
        return True, ""

    @classmethod
    def speaker_name(cls, lang: str, speaker: str | None) -> str | None:
        names = SPEAKERS.get(lang, ())
        if speaker and speaker in names:
            return speaker
        return names[0] if names else None

    @classmethod
    def version_for(cls, lang, speaker, options) -> str:
        return f"{cls._pins()[0].revision[:12]}:{cls.speaker_name(lang, speaker)}:{DESCRIPTION_REV}"

    def load(self, device: str) -> dict:
        from pathlib import Path

        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        from app.tts_runtime.manifest import model_dir, verify_locked

        home = Path(self.options.get("home") or runtime_home())
        pins = self._pins()
        for pin in pins:
            verify_locked(model_dir(home, pin), pin)          # ModelCorrupt -> never loaded
        main = model_dir(home, pins[0])
        self.device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(str(main)).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(main))
        self.desc_tokenizer = self.tokenizer if self.SHARED_TOKENIZER else \
            AutoTokenizer.from_pretrained(str(model_dir(home, pins[1])))
        self.sr = int(self.model.config.sampling_rate)
        return {"device": self.device, "sampling_rate": self.sr}

    def synth(self, text, lang, speaker, emotion, draw=0) -> SynthOutput:
        import torch

        name = self.speaker_name(lang, speaker)
        emo = EMOTIONS.get(emotion or "neutral") if lang in EMOTION_LANGS else None
        description = describe(name, emo)
        seed = int(hashlib.sha256(f"{draw}:{lang}:{name}:{text}".encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        d = self.desc_tokenizer(description, return_tensors="pt").to(self.device)
        p = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            gen = self.model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                      prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        x = gen.to(torch.float32).cpu().numpy().squeeze()
        return SynthOutput(x, self.sr, {"speaker": name, "description": description,
                                        "emotion_applied": bool(emo), "seed": seed})

    def health_check(self) -> dict:
        lang = "hi" if "hi" in self.languages else sorted(self.languages)[0]
        try:
            out = self.synth("नमस्ते", lang, None, None)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "detail": f"{type(e).__name__}: {e}"}
        import numpy as np

        return {"ok": bool(len(out.samples) > 0.1 * out.sr and np.isfinite(out.samples).all()),
                "detail": f"{len(out.samples) / out.sr:.2f}s"}

    def unload(self) -> None:
        self.model = None


class ParlerTinyAdapter(IndicParlerAdapter):
    """parler-tts-tiny-v1: same library and call path, English only, ungated.
    Exists so the Parler code path is tested for real without a token. v1
    models share one tokenizer for prompt and description (their model card)."""

    name = "parler_tiny"
    repo = "parler-tts/parler-tts-tiny-v1"
    license = "Apache-2.0"
    languages = frozenset({"en"})
    needs_token_for_setup = False
    PINS = ("parler_tiny",)
    SHARED_TOKENIZER = True

    @classmethod
    def speaker_name(cls, lang, speaker):
        return None

    def health_check(self) -> dict:
        try:
            out = self.synth("Hello there.", "en", None, None)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "detail": f"{type(e).__name__}: {e}"}
        return {"ok": len(out.samples) > 0.1 * out.sr, "detail": f"{len(out.samples) / out.sr:.2f}s"}
