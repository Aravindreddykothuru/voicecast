"""The licence allowlist. Nothing outside it is loaded -- checked when the
config is read and again when a model worker starts.

Only free, commercially usable, self-hosted weights. Every entry records
where its licence was verified, because "the model card says so" was not
enough once already: the Vakyansh readers carry no licence field on Hugging
Face at all, and are allowed only because their upstream repository is MIT.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass


class LicenseError(RuntimeError):
    """A model outside the allowlist, or explicitly forbidden, was asked for."""


@dataclass(frozen=True)
class Allowed:
    license: str                 # SPDX id
    repos: tuple[str, ...]       # glob patterns of the repositories this entry covers
    verified_at: str             # where the licence was checked
    kind: str = "tts"            # tts | reader
    test_only: bool = False      # usable by tests, never by a chain


ALLOWLIST: dict[str, Allowed] = {
    "syspin": Allowed(
        "CC-BY-4.0", ("SYSPIN/tts_vits_coquiai_*",),
        "huggingface.co/SYSPIN model cards (CC-BY-4.0, attribution required)"),
    "indic_parler": Allowed(
        "Apache-2.0", ("ai4bharat/indic-parler-tts", "google/flan-t5-large"),
        "huggingface.co/ai4bharat/indic-parler-tts card; flan-t5-large tokenizer Apache-2.0"),
    "indicf5": Allowed(
        "MIT", ("ai4bharat/IndicF5", "charactr/vocos-mel-24khz"),
        "huggingface.co/ai4bharat/IndicF5 card and github.com/AI4Bharat/IndicF5; its vocoder "
        "charactr/vocos-mel-24khz is MIT (card), fetched by IndicF5's load_vocoder()"),
    "vakyansh": Allowed(
        "MIT", ("Harveenchadha/vakyansh-wav2vec2-telugu-tem-100",
                "Harveenchadha/vakyansh-wav2vec2-kannada-knm-560",
                "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
                "Harveenchadha/vakyansh-wav2vec2-marathi-mrm-100",
                "Harveenchadha/vakyansh-wav2vec2-bengali-bnm-200"),
        "github.com/Open-Speech-EkStep/vakyansh-models (MIT; lists tem_100, knm_560, him_4200, "
        "mrm_100, bnm_200). The Hugging Face re-uploads have no licence field. hi/mr/bn are "
        "reliable on full sentences only, so they measure the benchmark but are not a short-line "
        "trigger (tts_chains.yaml readers: te, kn).", kind="reader"),
    "parler_tiny": Allowed(
        "Apache-2.0", ("parler-tts/parler-tts-tiny-v1",),
        "huggingface.co/parler-tts/parler-tts-tiny-v1 card. English only: exercises the "
        "Parler adapter code path in tests, never speaks a dub.", test_only=True),
    "fake": Allowed(
        "LicenseRef-no-weights", ("local/fake*",),
        "test double in app/tts_runtime/adapters/fake.py; loads no third-party weights",
        test_only=True),
}

# Named so that a config or adapter pointing at one fails with the reason,
# not just "not on the list".
FORBIDDEN: tuple[tuple[str, str], ...] = (
    (r"^facebook/mms-tts", "Meta MMS-TTS weights are CC-BY-NC-4.0 (non-commercial)"),
    (r"^SWivid/F5-TTS", "original F5-TTS weights are CC-BY-NC-4.0 (non-commercial)"),
    (r"(?i)xtts", "Coqui XTTS v2 is under the Coqui Public Model License (non-commercial)"),
)


def check(name: str, repo: str, license_id: str, *, allow_test_only: bool = False) -> Allowed:
    """The allowlist entry for model `name` loading `repo`, or LicenseError."""
    for pattern, why in FORBIDDEN:
        if re.search(pattern, repo):
            raise LicenseError(f"{name}: {repo} is forbidden -- {why}.")
    entry = ALLOWLIST.get(name)
    if entry is None:
        raise LicenseError(
            f"{name} is not on the licence allowlist (app/tts_runtime/licenses.py). Only free, "
            f"commercially licensed, self-hosted models may be loaded; add it there with where its "
            f"licence was verified, or remove it from tts_chains.yaml.")
    if entry.test_only and not allow_test_only:
        raise LicenseError(f"{name} is allowlisted for tests only and cannot be used by a chain.")
    if not any(fnmatch.fnmatchcase(repo, p) for p in entry.repos):
        raise LicenseError(f"{name}: repository {repo} is not covered by its allowlist entry {entry.repos}.")
    if license_id != entry.license:
        raise LicenseError(
            f"{name}: declares licence {license_id!r} but the allowlist records {entry.license!r} "
            f"({entry.verified_at}). Re-verify before changing either.")
    return entry
