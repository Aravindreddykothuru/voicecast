"""Vakyansh wav2vec2 CTC readers for the reader-check trigger (te, kn).

MIT per the upstream repository (licenses.py records why). Decoding is the
production gate's (scripts/e2e_dub.py::ctc_reader): greedy argmax, the
tokenizer's batch_decode, "<s>" stripped. These models emit <s> as the CTC
blank rather than their configured <pad>; collapsing repeats before dropping
it is exactly CTC decoding with <s> as blank.
"""
from __future__ import annotations

import numpy as np

from app.tts_runtime import licenses
from app.tts_runtime.adapters.base import Adapter

READER_REPOS = {
    "te": "Harveenchadha/vakyansh-wav2vec2-telugu-tem-100",
    "kn": "Harveenchadha/vakyansh-wav2vec2-kannada-knm-560",
    # Reliable on full sentences only (CONTRACTS.md #7): benchmark use, not a trigger.
    "hi": "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
    "mr": "Harveenchadha/vakyansh-wav2vec2-marathi-mrm-100",
    "bn": "Harveenchadha/vakyansh-wav2vec2-bengali-bnm-200",
}


class VakyanshReader(Adapter):
    name = "vakyansh"
    repo = READER_REPOS["te"]
    license = "MIT"
    languages = frozenset(READER_REPOS)
    kind = "reader"

    @classmethod
    def version_for(cls, lang, speaker, options) -> str:
        from app.tts_runtime.provision import READER_REVISIONS

        return READER_REVISIONS[READER_REPOS[lang]][:12]

    def load(self, device: str) -> dict:
        self.device = "cpu"
        self._models: dict[str, tuple] = {}
        return {"device": "cpu"}

    MAX_LOADED = 2     # ~380 MB each; five at once does not fit next to a TTS model on 16 GB

    def _model(self, lang: str):
        if lang in self._models:
            self._models[lang] = self._models.pop(lang)          # most recently used last
        else:
            while len(self._models) >= self.MAX_LOADED:
                self._models.pop(next(iter(self._models)))
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            from app.tts_runtime.provision import READER_REVISIONS

            repo = READER_REPOS[lang]
            licenses.check(self.name, repo, self.license)
            rev = READER_REVISIONS[repo]
            self._models[lang] = (Wav2Vec2Processor.from_pretrained(repo, revision=rev),
                                  Wav2Vec2ForCTC.from_pretrained(repo, revision=rev).eval())
        return self._models[lang]

    def read(self, samples: np.ndarray, sr: int, lang: str) -> str:
        import torch

        from app.tts_runtime.audio import as_mono_float, resample

        proc, model = self._model(lang)
        x = resample(as_mono_float(samples), sr, 16000)
        inputs = proc(x, sampling_rate=16000, return_tensors="pt").input_values
        with torch.inference_mode():
            ids = model(inputs).logits.argmax(-1)
        return proc.batch_decode(ids)[0].replace("<s>", "").strip()
