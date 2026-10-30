from __future__ import annotations

import logging

from app.config import get_settings, require_hf_token
from app.providers.base import TranslationProvider, TranslationResult
from app.providers.registry import ProviderNotInstalledError

logger = logging.getLogger(__name__)

# Deliberately NOT a table here: duplicating the language list is exactly how
# 5 of the 12 UI-offered languages ended up with no FLORES code and killed the
# job at translate time. Single source of truth: app/capabilities.py.
from app.capabilities import require_language, require_source_language


class IndicTrans2Provider(TranslationProvider):
    """AI4Bharat's IndicTrans2. Three separate checkpoints, one per direction
    (en->indic, indic->en, indic->indic) -- there is no single model that
    covers all pairs. Each is loaded lazily on first use, so a deployment
    that only ever dubs English sources never downloads the other two.

    Requires `pip install -r requirements-ml.txt`.
    """

    def __init__(self) -> None:
        try:
            import torch  # noqa: F401
            from IndicTransToolkit.processor import IndicProcessor  # noqa: F401
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401
        except ImportError as e:
            raise ProviderNotInstalledError("IndicTrans2Provider", "transformers, IndicTransToolkit") from e

        from IndicTransToolkit.processor import IndicProcessor

        self._processor = IndicProcessor(inference=True)
        # direction key ("en-indic" | "indic-en" | "indic-indic") -> (tokenizer, model)
        self._pairs: dict[str, tuple] = {}
        self._self_check()

    def _self_check(self) -> None:
        """Load the default direction and translate one fixed line now.

        Constructing this provider used to load no weights at all -- every
        direction was lazy -- so the worker's startup check (CONTRACTS.md #1)
        passed on a machine that could not translate, and the first job paid
        the ~4GB load and found out. English -> the default target is the
        path nearly every run takes; the other two directions stay lazy
        (a run that never sees an Indic source never downloads them) and
        still fail loudly on first use.
        """
        settings = get_settings()
        out = self.translate("Good morning.", settings.default_target_language, src_lang="en").text
        if not out.strip() or out.strip().lower() == "good morning.":
            from app.providers.loading import ModelLoadError

            raise ModelLoadError(f"IndicTrans2 self-check returned {out!r} for en->{settings.default_target_language}")
        logger.info("IndicTrans2 self-check passed (en->%s: %s)", settings.default_target_language, out)

    def _load_pair(self, direction: str, model_name: str):
        if direction in self._pairs:
            return self._pairs[direction]

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # All three IndicTrans2 checkpoints are "gated: auto" -- same account
        # that accepted pyannote's licence needs to accept these too. .env's
        # HF_TOKEN never becomes a real OS env var (pydantic-settings only
        # feeds it into Settings), so it's passed explicitly.
        token = require_hf_token(f"TRANSLATION_PROVIDER=real (IndicTrans2 {direction})")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, token=token)
        self._pairs[direction] = (tokenizer, model)
        logger.info("IndicTrans2: loaded %s direction (%s)", direction, model_name)
        return self._pairs[direction]

    def _direction_for(self, src_code: str, tgt_code: str) -> tuple[str, str]:
        settings = get_settings()
        if src_code == "en":
            return "en-indic", settings.translation_model_name
        if tgt_code == "en":
            return "indic-en", settings.translation_indic_en_model_name
        return "indic-indic", settings.translation_indic_indic_model_name

    def translate(self, text: str, target_lang: str, src_lang: str = "en") -> TranslationResult:
        # Both raise ValueError (a permanent, non-retryable failure -- see
        # app/pipeline/tasks.py's _PERMANENT tuple) rather than silently
        # treating an unsupported source as English. This is the fix for the
        # bug where ASR detected "zh" at 98% confidence and the pipeline
        # translated it through the en-indic model anyway, producing a
        # confident but wrong Telugu sentence with no error anywhere.
        src = require_source_language(src_lang)
        tgt = require_language(target_lang)

        direction, model_name = self._direction_for(src.code, tgt.code)
        tokenizer, model = self._load_pair(direction, model_name)

        import torch

        batch = self._processor.preprocess_batch([text], src_lang=src.flores, tgt_lang=tgt.flores)
        inputs = tokenizer(batch, return_tensors="pt", padding="longest", truncation=True)
        # Beam search as in AI4Bharat's reference inference; greedy decoding
        # (the previous default) drops or repeats words on longer lines.
        with torch.inference_mode():
            generated = model.generate(**inputs, max_length=256, num_beams=5, num_return_sequences=1)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        out = self._processor.postprocess_batch(decoded, lang=tgt.flores)[0]
        return TranslationResult(text=out, src_lang=src_lang, target_lang=target_lang)
