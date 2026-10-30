# Licence register

Every model this project loads, with the licence read from the Hugging Face
API **at the pinned revision** — not from memory and not from a blog post.
`app/tts_runtime/licenses.py` is the machine-readable allowlist and is
checked when the config loads and again inside every worker; this file is
the human record of where each licence was verified.

Last verified **2026-10-01** against the revisions below
(`docs/tts-preflight.json`).

| model | repo | revision | licence | production |
|---|---|---|---|---|
| syspin | `SYSPIN/tts_vits_coquiai_*` | pinned + sha256 per file | CC-BY-4.0 | **yes** — attribution to "SYSPIN, IISc Bangalore and ARTPARK" |
| indic_parler | `ai4bharat/indic-parler-tts` | `7b527af5ee8e` | Apache-2.0 | **yes**, once benchmarked |
| indicf5 | `ai4bharat/IndicF5` | `ba85abedf18d` | MIT | **no** — disabled pending a consented reference clip |
| indicf5 vocoder | `charactr/vocos-mel-24khz` | pinned | MIT | with IndicF5 |
| indic_conformer | `ai4bharat/indic-conformer-600m-multilingual` | `e9b71b369c04` | MIT | **yes** as a reader, once calibrated |
| vakyansh (reader) | `Harveenchadha/vakyansh-wav2vec2-*` | pinned | MIT (upstream repo; the HF re-uploads carry no licence field) | **yes** — te/kn as a trigger, hi/mr/bn benchmark-only |
| whisper large-v3 (reader) | `openai/whisper-large-v3` via faster-whisper | — | MIT | **yes** as a reader |
| parler_tiny | `parler-tts/parler-tts-tiny-v1` | pinned | Apache-2.0 | **tests only** — English, never speaks a dub |

## Refused by name

`licenses.FORBIDDEN` rejects these wherever they appear, with the reason:

| pattern | why |
|---|---|
| `^facebook/mms-tts` | Meta MMS-TTS is CC-BY-NC-4.0 — non-commercial |
| `^SWivid/F5-TTS` | original F5-TTS weights are CC-BY-NC-4.0 |
| `(?i)xtts` | Coqui XTTS v2, Coqui Public Model License — non-commercial |

The MMS family is **benchmark-only** and was measured outside the runtime so
no chain could ever select it (`docs/tts-alternatives.md`). The same bar
applies to readers: `facebook/mms-1b-all` is MMS and is therefore not a
candidate, which is why the reader comparison uses Whisper.

## The rule

Non-commercial means benchmark-only: it may be measured, it may be written
about, it never enters a production chain. Anything not on the allowlist is
refused rather than assumed permissive, and a model whose declared licence
differs from the recorded one is refused until someone re-verifies both.
