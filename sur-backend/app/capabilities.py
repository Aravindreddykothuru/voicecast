"""The single source of truth for what this backend can actually do.

Why this exists: the frontend used to carry its own hand-written list of 12
target languages. The translation provider only had FLORES codes for 7 of
them, so picking Gujarati, Punjabi, Odia, Assamese or Urdu raised ValueError
deep in the translate stage and killed the whole job. Two lists, one of them
wrong, no test connecting them.

Nothing may hardcode a language list again. The frontend renders whatever
GET /api/capabilities returns; the translation provider derives its FLORES
mapping from here; a test asserts every entry is complete.

See CONTRACTS.md, invariant #2.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    """One TTS checkpoint that can speak a language.

    License is part of the capability, not a footnote: a deployment that
    must be commercially usable (TTS_REQUIRE_COMMERCIAL_LICENSE=true) may
    only offer languages that have a voice with `commercial=True`, and the
    UI shows those as unavailable rather than letting a job run on weights
    the business may not ship. See CONTRACTS.md #2 and #7.
    """

    engine: str             # "mms" | "syspin" -- which provider code loads it
    model: str              # Hugging Face repo id
    license: str            # SPDX-style identifier of the weights' license
    commercial: bool        # whether the license permits commercial use
    gender: str | None = None


def _mms(suffix: str) -> Voice:
    # facebook/mms-tts-*: CC-BY-NC-4.0 -- research/non-commercial only.
    return Voice("mms", f"facebook/mms-tts-{suffix}", "CC-BY-NC-4.0", commercial=False)


def _syspin(name: str, gender: str) -> Voice:
    # IISc SYSPIN VITS (TorchScript): CC-BY-4.0 -- commercial use allowed
    # with attribution. Character-based, so no GPL phonemizer is involved.
    return Voice("syspin", f"SYSPIN/tts_vits_coquiai_{name}{gender.capitalize()}", "CC-BY-4.0", commercial=True, gender=gender)


@dataclass(frozen=True)
class Language:
    code: str          # what the API and DB speak, e.g. "te"
    name: str          # what the UI shows, e.g. "Telugu"
    flores: str        # what IndicTrans2 needs, e.g. "tel_Telu"
    # Every checkpoint that can speak this language. Availability is derived
    # from these (tts_voices), never asserted: every row used to say
    # tts_supported=True while the configured TTS model (CosyVoice2) could
    # not speak a single one of these languages.
    voices: tuple[Voice, ...] = ()
    # Scaffolding, never shipped: SYSPIN renders a line that is the whole
    # utterance as a *different word* -- 8 one-word lines measured 2/8
    # intelligible alone against 8/8 when spoken after a carrier and cut back
    # out (issue #5, CONTRACTS.md #7). A short line is rendered after this
    # sentence and excised. Set only for languages where that has been
    # measured; None leaves the line rendered on its own, as before.
    tts_carrier: str | None = None
    # Who checked that the carrier is natural, idiomatic text in this
    # language. "machine" means only a round-trip check was done -- the
    # phrase is machine-translated and back-translates to the intended
    # meaning, and nothing more than that is claimed. A carrier is never
    # heard by a viewer, so an awkward one is not a product defect, but it
    # does drive the engine, and an ungrammatical phrase would render
    # oddly into the line it scaffolds. Set to a person and a date when a
    # fluent speaker has actually read it.
    tts_carrier_review: str = "machine: back-translated, no human review"
    # Whether the carrier cut may walk back across a within-word silence (a
    # stop closure) rather than start the line mid-word -- common.
    # tail_after_pause. Measured per carrier against forced-alignment ground
    # truth: safe for hi, kn, mr and bn (over 640 renders it added no carrier
    # leak, moved no cut that was already at the boundary, and took mid-word
    # cuts from 69 to 1); unsafe for te, whose carrier ends on a short word
    # that the merge swept into the clip in 13 of 407 held-out renders.
    tts_carrier_merge_closures: bool = False

    @property
    def tts_supported(self) -> bool:
        return bool(self.voices)


# Every entry MUST carry a FLORES code: an entry without one is a language the
# UI would offer and the pipeline would then fail on. tests/test_capabilities.py
# enforces that, so adding a row without a mapping fails the build.
# MMS suffixes verified to exist on the Hub, none requiring uroman
# pre-processing (tokenizer_config.is_uroman == false). Urdu's checkpoint is
# script-qualified; plain "urd" does not exist. SYSPIN voices are the
# TorchScript releases; its Gujarati release is a Coqui checkpoint that needs
# the Coqui runtime and is not wired up.
SUPPORTED_LANGUAGES: tuple[Language, ...] = (
    Language("hi", "Hindi", "hin_Deva", (_mms("hin"), _syspin("Hindi", "male"), _syspin("Hindi", "female")),
             tts_carrier="यह एक भयानक खबर है।", tts_carrier_merge_closures=True),
    Language("te", "Telugu", "tel_Telu", (_mms("tel"), _syspin("Telugu", "male"), _syspin("Telugu", "female")),
             tts_carrier="అది భయంకరమైన వార్త."),
    Language("ta", "Tamil", "tam_Taml", (_mms("tam"),)),
    Language("kn", "Kannada", "kan_Knda", (_mms("kan"), _syspin("Kannada", "male"), _syspin("Kannada", "female")),
             tts_carrier="ಇದು ಭಯಾನಕ ಸುದ್ದಿ.", tts_carrier_merge_closures=True),
    Language("ml", "Malayalam", "mal_Mlym", (_mms("mal"),)),
    Language("bn", "Bengali", "ben_Beng", (_mms("ben"), _syspin("Bengali", "male"), _syspin("Bengali", "female")),
             tts_carrier="এটা একটা ভয়ংকর খবর।", tts_carrier_merge_closures=True),
    Language("mr", "Marathi", "mar_Deva", (_mms("mar"), _syspin("Marathi", "male"), _syspin("Marathi", "female")),
             tts_carrier="ही भीतीदायक बातमी आहे.", tts_carrier_merge_closures=True),
    Language("gu", "Gujarati", "guj_Gujr", (_mms("guj"),)),
    Language("pa", "Punjabi", "pan_Guru", (_mms("pan"),)),
    Language("or", "Odia", "ory_Orya", (_mms("ory"),)),
    Language("as", "Assamese", "asm_Beng", (_mms("asm"),)),
    Language("ur", "Urdu", "urd_Arab", (_mms("urd-script_arabic"),)),
)


def tts_voices(lang: Language) -> list[Voice]:
    """The voices the configured engine may use for `lang`, license policy applied."""
    from app.config import get_settings

    settings = get_settings()
    return [
        v for v in lang.voices
        if v.engine == settings.tts_engine and (v.commercial or not settings.tts_require_commercial_license)
    ]


def tts_available(lang: Language) -> bool:
    """Whether a dub INTO `lang` can be rendered on this deployment. The mock
    provider speaks anything; a real engine needs a voice that passes policy."""
    from app.config import get_settings

    return get_settings().tts_provider == "mock" or bool(tts_voices(lang))

LANGUAGES_BY_CODE: dict[str, Language] = {lang.code: lang for lang in SUPPORTED_LANGUAGES}

# Derived, never hand-maintained: the translation provider reads this.
FLORES_CODES: dict[str, str] = {lang.code: lang.flores for lang in SUPPORTED_LANGUAGES}


def require_language(code: str) -> Language:
    """Resolve a target (dub-into) language or fail with the full supported set.

    Deliberately raises instead of falling back to a default: silently dubbing
    into the wrong language is worse than refusing. See CONTRACTS.md #3.
    """
    return _require(code, LANGUAGES_BY_CODE, "target")


# Source (transcribe-from) languages this engine can correctly translate.
# English uses IndicTrans2's en-indic direction; every Indic language below
# round-trips through indic-en / indic-indic. This is NOT the same set as
# SUPPORTED_LANGUAGES conceptually (that one is "what can we dub into"), it
# just happens to be English + the identical Indic list, because that's what
# ai4bharat/indictrans2-{indic-en,indic-indic} actually cover.
#
# A source language NOT in this set (the original bug: ASR detected "zh" at
# 98% confidence, and translation silently ran it through the en-indic model
# anyway, producing confident nonsense) must be refused, not guessed. See
# CONTRACTS.md #3 and #5.
SOURCE_LANGUAGES: tuple[Language, ...] = (
    Language("en", "English", "eng_Latn"),
) + SUPPORTED_LANGUAGES

SOURCE_LANGUAGES_BY_CODE: dict[str, Language] = {lang.code: lang for lang in SOURCE_LANGUAGES}


def require_source_language(code: str) -> Language:
    """Resolve a source (transcribed-from) language or fail with the full
    supported set. See CONTRACTS.md #3 -- this is the guard the original bug
    had none of: ASR can detect (and did detect) languages this pipeline has
    no correct translation path for, and that must stop the run rather than
    silently mistranslate.
    """
    return _require(code, SOURCE_LANGUAGES_BY_CODE, "source")


def _require(code: str, table: dict[str, Language], kind: str) -> Language:
    lang = table.get(code)
    if lang is None:
        raise ValueError(
            f"Unsupported {kind} language {code!r}. Supported {kind} languages: "
            f"{', '.join(sorted(table))}. "
            f"Add it to app/capabilities.{'SOURCE_LANGUAGES' if kind == 'source' else 'SUPPORTED_LANGUAGES'} "
            "(with its FLORES code and a real translation path) rather than special-casing it here."
        )
    if not lang.flores:
        # Checked on every resolution, not just in the real translation
        # provider: otherwise a half-filled entry sails through mocked runs
        # (which don't translate for real) and only detonates in production.
        raise ValueError(
            f"{kind.capitalize()} language {code!r} ({lang.name}) is advertised but has no "
            f"FLORES code, so translation would fail at run time. Complete the entry."
        )
    return lang


# Colour per canonical emotion label. Lives here, not in the frontend, so the
# UI never has to keep its own map in sync with what the model emits -- a
# label the backend adds arrives with its colour already decided.
# Values are the palette the editor already uses.
EMOTION_COLORS: dict[str, str] = {
    "neutral": "#8898c8",
    "happiness": "#34d399",
    "anger": "#f87171",
    "sadness": "#818cf8",
    "fear": "#a78bfa",
    "surprise": "#fbbf24",
}
EMOTION_COLOR_FALLBACK = "#8898c8"
