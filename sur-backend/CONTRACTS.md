# Contracts

Invariants this codebase must preserve. Each one exists because it was
violated in production and nothing failed loudly enough to notice.

Read this before changing model loading, capability lists, the mux stage, or
secret handling. Every invariant is enforced by a test named in its section —
if you find yourself deleting or weakening that test, you are about to
reintroduce a shipped bug.

---

## 1. A model load either brings its weights or fails

**What went wrong.** `transformers.from_pretrained` treats a checkpoint whose
keys don't match the architecture as a *success*: it logs a warning and
silently random-initialises whatever it couldn't fill in. The emotion
classifier (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`)
stores its head as `classifier.dense`/`classifier.output`, while the current
`Wav2Vec2ForSequenceClassification` wants `projector.*`/`classifier.*`. So the
model ran on a **random head** for an entire release, emitting labels at
chance confidence (0.14 across 8 labels). The API returned 200. Nothing
alerted. The only symptom was that the emotion labels were meaningless.

**The rule.**
- No production code calls `from_pretrained` directly for a model with
  weights. All loads go through `app/providers/loading.py::load_hf_model`.
- Any missing weight that is not on the explicit benign allowlist is a hard
  `ModelLoadError`. The allowlist contains exactly one rule: the PyTorch
  `weight_norm` rename (`X.weight_g`/`X.weight_v` →
  `X.parametrizations.weight.original0/1`), which is a spelling change, not
  absent training -- and it is applied **pairwise**: a missing
  `X.parametrizations.weight.original0` is excused only if the checkpoint
  actually carries `X.weight_g`. (It used to be a regex for wav2vec2's one
  conv layer; MMS-TTS reports the rename on 128 keys, and a pattern broad
  enough for those would also excuse a genuinely absent layer.)
- Every real provider's constructor loads the weights its common path uses
  and runs a behavioural self-check. The translation provider used to load
  nothing until the first job, so its startup check proved nothing.
- Missing **head** weights (`classifier`, `projector`, `score`, `head`,
  `out_proj`) are never tolerated, allowlist or not.
- Every provider with a classification head runs `assert_not_degenerate` on a
  fixed built-in sample at construction. A near-uniform output distribution
  is treated as a failure to boot, whatever its cause.
- Workers run these checks at startup (`app/startup_checks.py`) so a bad
  deployment dies immediately instead of failing customer jobs one at a time.
- **SYSPIN voices load only at a pinned commit, and every file they use is
  hashed first** (`app/providers/tts/syspin_manifest.py`). Unpinned: a
  `ModelLoadError` naming the manifest. One changed byte: a `ModelLoadError`
  naming the voice, revision, file and both hashes. A release that moves
  upstream therefore cannot change the dub's speech without a code change.
  `SYSPIN/tts_vits_coquiai_BengaliFemale` was published without `extra.py`
  (issue #6); that file is byte-identical across the nine releases that ship
  it, so the voice borrows the pinned copy from a sibling release, verifies
  its hash, logs `SUPPLY CHAIN` at WARNING on every load, and is listed in
  `/api/capabilities` under `tts_voice_warnings` -- worked around, never
  hidden. An upstream report is drafted in
  `docs/upstream/syspin-bengalifemale-missing-extra-py.md`.

**Enforced by.** `tests/test_model_loading_is_fail_loud.py`,
`tests/test_startup_checks.py`, `tests/test_voice_supply_chain.py` (every
advertised voice pinned, well-formed pins, unpinned and tampered loads
refused with a precise message, the borrowed file fetched at a pinned
commit, verified and logged, the Bengali gap reported). The first test is
the exact ehcalabres key
signature; the second asserts the good model (which reports the benign pair
as missing) still loads, so the fix can't be "reject everything".

---

## 2. The UI cannot offer what the backend cannot serve

**What went wrong.** The frontend carried a hand-written list of 12 target
languages. The translation provider had FLORES codes for 7. Choosing
Gujarati, Punjabi, Odia, Assamese or Urdu raised `ValueError` deep inside the
translate stage and killed the whole project. Two lists, one of them wrong,
and no test connecting them. Separately, the UI rendered six emotion buttons
while the model could only ever predict four.

**The rule.**
- `app/capabilities.py::SUPPORTED_LANGUAGES` is the single source of truth for
  languages: code, display name, FLORES code, TTS availability. Nothing else
  may define a language list — not the providers, not the frontend.
- Translation providers derive their FLORES mapping from it. `require_language`
  is the only way to resolve a target language, and it validates completeness
  on every call, so a half-filled entry fails in mocked runs too rather than
  waiting for production.
- TTS availability per language is derived from the voices that speak it
  (`Language.voices`, filtered by the configured engine and license policy in
  `tts_voices`), never a bare `True`. Every row used to say
  `tts_supported=True` while the configured TTS model (CosyVoice2) could not
  speak any of the twelve languages. `POST /api/projects` refuses a target
  language with no voice (422) instead of failing at the synthesize stage.
- `voice_clone_available` is published and `/process` enforces it; a target
  language is validated when the project is created, not after upload,
  extraction and ASR have run.
- Emotion labels are derived from the loaded model's own `config.id2label`
  (`EmotionProvider.available_labels()`), never hardcoded.
- Both are published at `GET /api/capabilities`. The frontend renders that
  response and holds no lists of its own.

**Enforced by.** `tests/test_capabilities.py` (every entry has a FLORES code,
codes are unique, the endpoint matches the table, emotion labels come from the
provider) and `tests/test_pipeline_contract_all_languages.py`, which runs the
real orchestration end-to-end for **every** advertised language — the original
bug was that 5 of them had never been executed once.

---

## 3. Timeline accuracy: speech stays under the mouth it belongs to

**What went wrong.** `mux_export` concatenated the rendered clips back to
back, discarding every segment's `start_ms` and every gap between utterances.
The dub slid progressively out of sync with the picture. The output was a
valid video of roughly the right length, so nothing failed — you had to
listen to it to find out.

A second, subtler version of the same bug: `-shortest` without padding
truncated the *video* to wherever the audio ended, so a 10s clip whose last
line landed at 6s came out 6s long.

**The rule.**
- `mux_timeline` places each clip at its own `start_ms` (`adelay`), mixes them
  (`amix`), and pads the result (`apad`) so the picture is never truncated.
- Placement keys off `start_ms`, never list position.
- Output duration must equal source duration.
- Segments never overlap. Raw diarization turns overlap and fragment;
  `timeline.normalize_turns` resolves overlaps (the later speaker keeps its
  start), merges same-speaker fragments, and absorbs blips before any
  segment row exists.
- A clip longer than the time before the next spoken line is sped up
  (pitch-preserving `atempo`) just enough to fit, capped at `MAX_TEMPO`;
  whatever still doesn't fit is reported in the QA report (`overrun_ms`),
  never trimmed silently. Translated Indic speech is routinely longer than
  the English it replaces, so "placed at start_ms" alone still produced
  one line talking over the next. See `timeline.plan_timeline`.
- Segments with no speech produce no clip; their stretch stays silent.

**Enforced by.** `tests/test_timeline_planning.py` (pure planning, including
the verbatim pyannote output that motivated it) and
`tests/test_timeline_accuracy.py`. The latter render with
**real ffmpeg** and then measure actual audio energy per time window — they
assert energy at each expected timestamp and silence in the gaps. Mocking
ffmpeg here would test nothing, because the bug lived in the ffmpeg
invocation. Re-introducing concatenation fails three of them.

CI must have ffmpeg installed: `test_ffmpeg_is_available_in_ci` fails when
`CI` is set and ffmpeg is missing, so the suite can't silently skip the only
guard on this invariant.

---

## 4. Secrets live in the environment, not in the repo

**What went wrong.** The Hugging Face token sat in `.env` on disk. A
long-lived credential in a file gets copied, committed, and shared, and
rotating it means editing and redeploying.

**The rule.**
- `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) is read from the process
  environment at call time — `app/config.py::hf_token`. It is deliberately
  **not** a `Settings` field, so it is never read from `.env` and never cached.
- Callers that need it use `require_hf_token(reason)`, which raises a message
  naming the variable and the provider instead of letting a gated download
  return an opaque 401.
- Workers verify it at startup when a gated provider is set to `real`.
- Exception: under `HF_HUB_OFFLINE=1` (a process env var, read by
  huggingface_hub itself) nothing is downloaded, so no token is required.
  The gated weights must already be cached, and the startup model load
  proves they are.

**Enforced by.** `tests/test_secrets.py` — no token value in any `.env`, no
token literal anywhere in tracked source, the accessor raises actionably when
unset, and the value is re-read on each call so rotation is a restart.
`tests/test_pipeline_hardening.py` covers the offline exception.

---

## 5. No silent defaults

Cutting across all of the above: when a required value is absent, raise. Do
not substitute `"en"`, `0`, `None`, or the first plausible entry.

Concretely: ASR source language defaults to `auto` (detect) rather than
assuming English — it previously hardcoded `language="en"`, so every
non-English source was decoded as English. `require_language` raises rather
than falling back to a default target. Permanent failures (`ValueError`,
`LookupError`, `TypeError`) are excluded from Celery autoretry via
`dont_autoretry_for`, because retrying "project not found" three times with
backoff only delays the error an operator needs to see.

More of the same, found running the real pipeline end to end:
`probe_duration_ms` swallowed every ffprobe error and returned 0;
`LocalStorage.download_file` returned silently for a missing key; a missing
ffmpeg binary surfaced as "exit status 1" with no stderr; a
`/process` with `review_language=false` queued only half the chain and
stranded the project in "processing"; the UI's pinned source language was
dropped by a 409 nobody saw. Each now raises or is honoured, with a test in
`tests/test_pipeline_hardening.py`.

---

## 6. Long stages are resumable, and never run twice at once

**What went wrong.** `synthesize` rendered every segment inside one database
transaction. On CPU that is minutes per segment; a failure on the last one
rolled back every finished render and the retry started from zero. Separately,
with `task_acks_late` Redis re-delivers any unacked message after its
visibility timeout (default one hour) -- to another worker, while the first is
still running it.

**The rule.**
- Per-segment stages commit each segment as it finishes; a retry resumes from
  the first unfinished one.
- `broker_transport_options.visibility_timeout` exceeds the longest stage.
- A task runs only on a worker whose venv has its models: `regenerate` is
  split into a translate half (`q.translate`) and a synthesize half
  (`q.synthesize`), because the TTS venv has no IndicTrans2.

**Enforced by.** `tests/test_pipeline_hardening.py`.

---

## 7. Speech comes from a voice we may ship, is fitted once, and is cloned only within budget

**What went wrong.**
- The only working Indic engine, MMS-TTS, is CC-BY-NC-4.0, so every dub was
  unshippable, and nothing in the code knew that.
- Voice cloning (CosyVoice2 voice conversion) raised the dub's Whisper CER
  from 0.16 to 0.24, reaching 0.44 on one line, and took 12.5 CPU-minutes
  for 30 s of speech. Nothing measured or bounded either cost.
- One-word lines read back badly. The engine sped a line up to 1.25× to fit
  its window, then `plan_timeline` sped the already fitted clip up to
  another 1.35×. That is 1.69× and two WSOLA passes on exactly the lines with
  the least audio to spare. The QA report's `tempo` showed only the second
  factor.
- An earlier emotion pitch shift blurred short lines (CER 0.21 → 0.36).

**The rule.**
- Every voice is declared in `app/capabilities.py` with its engine, model,
  license and commercial flag. `TTS_ENGINE` selects the engine, so a swap is
  a config change behind the same `TTSProvider` interface.
- With `TTS_REQUIRE_COMMERCIAL_LICENSE=true` (the default):
  - only commercially licensed voices are offered
  - the registry refuses to build a non-commercial engine
  - a language with no such voice reports `tts_available=false`
  - `POST /api/projects` returns 422 for that language
  - `/api/capabilities` publishes `tts_engine`, `tts_licenses` and
    `tts_commercial_use`
- Default engine: **SYSPIN VITS** (IISc/ARTPARK, **CC-BY-4.0**; commercial use
  with attribution to "SYSPIN, IISc Bangalore and ARTPARK"). It covers te,
  hi, kn, mr and bn with male and female voices. Each speaker gets the voice
  matching their estimated F0 (`speakers.voice_gender`).
- **Engines never time-fit.** `mux_export` (`timeline.plan_timeline`) is the
  single place a clip is sped up, so its `tempo` (≤ `MAX_TEMPO`) is the
  whole compression and the QA report shows it. An engine applies an
  emotion's rate only when the line fits its window, and renders at neutral
  rate otherwise.
- **A line always renders the same way, and short lines are drawn more than
  once.** Every engine seeds its sampler from the text, so re-rendering one
  unchanged segment (`/regenerate`) returns the same clip rather than a new
  one of a different length. MMS seeded per text from the start; SYSPIN did
  not, and drew a 0.21 s word in a real run.
  On top of that, a line at or below `SHORT_LINE_CHARS` is drawn
  `SHORT_LINE_DRAWS` times and the median by duration kept
  (`common.render_stable`): the duration head's spread explodes on short
  input and the rushed tail of it is unintelligible (see *Measured*). Wired
  into SYSPIN, the default; MMS still renders once per line, and is refused
  outright while `TTS_REQUIRE_COMMERCIAL_LICENSE` is on.
- **Voice cloning must pass a budget before the TTS worker accepts work.**
  `voice_clone.check_budget` converts a fixed probe and checks:
  - real-time factor ≤ `TTS_VOICE_CLONE_MAX_RTF`, taken as the fastest of up
    to three timed draws (contention only ever adds time, and a single sample
    measured 4.65x for a converter that runs at 2.2x -- which refused to start
    a worker that also serves plain, uncloned synthesis)
  - duration within 10%
  - speech envelope preserved (correlation ≥ 0.6)

  A converter that fails is a startup error, not a slow or garbled job.
  Default converter: OpenVoice V2 (MIT). `/process` refuses `clone_voice`
  when cloning is unavailable.
- Emotion is rendered only through controls the model really has:
  - SYSPIN's TorchScript graph takes token ids only, so emotion there is
    post-hoc rate (pitch-preserving) and energy
  - MMS adds VITS `noise_scale`
  - there is no pitch shift and no emotion conditioning, and the code says
    so rather than pretending otherwise

**Measured** (Whisper large-v3 CER, Telugu, forced language):

All on the same 32.56 s two-speaker English clip (9 lines) dubbed to Telugu,
read back with `scripts/e2e_dub.py --asr-check` on an 8-core CPU box.

| Engine / path | Corpus CER | Notes |
|---|---|---|
| SYSPIN VITS (default, CC-BY-4.0) | **0.194** | full sentences 0.11-0.25; 2026-09-20 |
| MMS-TTS (CC-BY-NC-4.0) | 0.160 | 2026-09-13, before SYSPIN was the default |
| MMS + CosyVoice2 VC (cloned) | 0.240 | worst 3-word line 0.44; 2026-09-13 |
| CosyVoice2 as the *renderer* | 1.56 | never trained on an Indic script; 30+ CPU-minutes for one 3 s line |

Converter speed, 3 s probe, real-time factor (`check_budget`):

| Converter | RTF | |
|---|---|---|
| OpenVoice V2 (default, MIT) | 1.9-3.3 | 4.65 measured while another worker was loading a model, which is why the check keeps the fastest of a few draws |
| CosyVoice2 VC | ~25 | 12.5 CPU-minutes for 30 s of speech |

Cloned CER for the current default pair (SYSPIN + OpenVoice V2) has not been
measured; the 0.240 above is the older MMS + CosyVoice2 pair.

**Short lines are where this engine fails.** VITS samples each token's
duration, and the relative spread grows as the line shortens: ten draws of a
one-word Telugu line ("సరే.", *okay*) spanned 0.325-0.836 s (x2.57), against
x1.18 for a 19-character line. A run that drew 0.21 s for that word produced
audio Whisper read as a different word entirely, and the clip was time-
compressed on top (mux tempo 1.115). Drawing short lines three times and
keeping the median removed both: that line renders at 0.33-0.44 s, the run's
max tempo fell to 1.0, and a two-word line that had scored CER 0.50 scored
0.083. A 0.33 s interjection still sits below what Whisper
recognises, and scoring it inside the surrounding dub does not rescue it when
the line has silence on both sides.

**That is the recogniser. There was also a real defect underneath it.**
SYSPIN renders a line that is the *entire utterance* as a different word.
Measured over 8 words x 5 draws x 4 layouts, CTC read-back:

| the line is | median CER | intelligible |
|---|---|---|
| the whole utterance | 0.775 | 7/40 |
| first, with a sentence after it | 0.000 | 35/40 |
| last, after a sentence | 0.000 | 39/40 |
| in the middle of a sentence | 0.000 | 37/40 |

Not tokenization (the ids for the word are identical in every layout), not
`trim_silence` (the untrimmed render fails too), not the emotion rate, and
not duration -- a 380 ms word inside a carrier reads back while a 360 ms one
alone does not. Any carrier fixes it.

**So a short line is spoken after a carrier sentence and cut back out**
(`common.render_line`, `Language.tts_carrier`). The clip that ships is still
only the line at its natural length; the carrier is scaffolding and is never
heard. On lines the pipeline's own translator produced from real short source
lines, median of 3 draws, short lines passing the 0.35 bar:

| | te | hi | kn | mr | bn |
|---|---|---|---|---|---|
| rendered alone | 5/9 | 1/8 | 5/8 | 2/8 | 1/8 |
| after a carrier, excised | 9/9 | 6/8 | 8/8 | 6/8 | 7/8 |
| same, re-measured 2026-09-22 (merge off) | 9/9 | 7/8 | 8/8 | 6/8 | 7/8 |
| **+ closure merge (below)** | **9/9** | **7/8** | **8/8** | **6/8** | **8/8** |

The last two rows cut the *same* renders, so they differ only by the merge:
no line got worse, and `সত্যিই` moved from CER 0.667 (`তি`) to 0.333 -- a
pass, but a narrow one. Hindi's 6/8 -> 7/8 between the first two rows is not
this change; it is the code between that measurement and this one.

Lengthening or padding a render to clear the bar remains forbidden -- this
changes *how* the engine is driven, not how long the output is.

**Where the cut loses words, measured at N = 200 per line in two sets.** An
earlier reading blamed the cut for `रुकिए` and `क्यों`. Scoring each line
*in place* -- the whole carrier render read by CTC and aligned to the
expected text, with no cut at all -- against the shipped clip, over 200 fresh
draws in each of two independent sets, says otherwise. `रुकिए` reads the same
either way (in place 54.5% / 58.5%, shipped 55.5% / 58.0%): the cut loses
nothing there; the render itself is misheard about 40% of the time. `क्यों`
has at most ~8 points of headroom (75.0% / 68.0% against 67.0% / 70.0%), and
none in the second set.

Acoustic boundary snapping (extend the tail until the energy stays under the
measured noise floor for 30 ms, move the head to the preceding energy
minimum, cut on zero crossings, 5 ms fades) made both focus lines worse --
`रुकिए` 55.5 -> 44.0% and 58.0 -> 47.0% (exact McNemar p = 0.0001, 0.0007),
`क्यों` unchanged (p = 0.44, 0.28) -- and regressed three other lines
(`হ্যাঁ` 65 -> 27.5%, `का` 82.5 -> 62.5%, `ఇప్పుడే` 95 -> 60%, N = 40 each).
Ablated, the snapped head does the damage and the snapped tail is neutral.
Not shipped.

The losses the cut does cause are elsewhere. Forced-alignment ground truth
over 814 renders showed the silence splitter taking a stop consonant's
closure *inside* the line for the pause before it (the `ত্` of `সত্যিই`),
so the clip began mid-word in 92 of 814 renders (11%), and `সত্যিই` -- read
back as `তি` -- passed in 0 of the 440 draws measured. Such gaps are at most
116 ms with at most 279 ms of speech before them; the carrier's last segment
is a median 1068 ms. So the cut now walks back across a gap under 140 ms
that follows under 300 ms of speech
(`common.CLOSURE_GAP_MS`, `Language.tts_carrier_merge_closures`). Confirmed on
fresh draws, 200 per line per set, pass = CER <= 0.35:

| line | set | before | after | paired: better / worse |
|---|---|---|---|---|
| `সত্যিই` | A' | 0.0% [0.0, 1.9] | 59.0% [52.1, 65.6] | 118 / 0 |
| `সত্যিই` | B' | 0.0% [0.0, 1.9] | 67.5% [60.7, 73.6] | 135 / 0 |
| `এখনই` | A' | 54.0% [47.1, 60.8] | 86.0% [80.5, 90.1] | 64 / 0 |
| `এখনই` | B' | 56.5% [49.6, 63.2] | 88.0% [82.8, 91.8] | 63 / 0 |

Over all 1000 paired held-out clips on hi/kn/mr/bn (every line; N = 20 each,
200 for `क्यों` and `रुकिए`) no clip got worse and 30 got better (p = 2e-9).
`क्यों` is unchanged (140/200 both ways) and `रुकिए` moved 116 -> 120 (p =
0.13) -- as the in-place numbers predicted, the cut was not what failed
them. It is off for Telugu: its carrier ends on a short word followed by a
short pause, and the rule swept that word into the clip in 13 of 407
held-out renders.

Each carrier is machine-translated and back-translates through an independent
model to the intended meaning, which is all `Language.tts_carrier_review`
claims. None has been read by a fluent speaker. A one-page sheet per language
-- the carrier, its NLLB back-translation, the carrier spoken, and a short
line as it ships -- is in `docs/carrier-review/review-packet.html` for one to
fill in (built by nothing in the pipeline; regenerate it by hand if a carrier
changes). Marathi is first: its carrier back-translates as "frightening
news", not "terrible news". `tts_carrier_review` changes only when a signed
sheet comes back.

**So the gate stops asking Whisper, and asks something that can answer.**
Fifteen clips padded with silence exactly as the dub places a line, scored
the way the gate scores a short line:

| ms | 372 | 418 | 511 | 534 | 627 | 673 | 766 | 789 | 998 | 1022 | 1022 | 1091 | 1207 | 1231 | 1602 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CER | 1.00 | 0.50 | 0.25 | 1.33 | 0.43 | 0.83 | 0.80 | 0.50 | 0.17 | 0.40 | 0.86 | 0.75 | **0.08** | **0.25** | **0.31** |

Whisper is autoregressive and needs an utterance to condition on; below
`ASR_FLOOR_MS` (1200 ms) it is not asked. A frame-synchronous CTC recogniser
has no length prior and reads complete words of 360-1060 ms alone, exact on 7
of 8, so that is what scores a short line -- against the same 0.35 bar, and
only for a language with a model in `SHORT_LINE_CTC_MODELS`.

**A reader earns its place by passing correct lines, not by reading long ones
well.** te and hi/kn/mr/bn were all listed at first, on the strength of their
accuracy on long renders (median CER 0.02-0.12). That was the wrong test: on
the short lines the gate actually scores, the readers for hi, mr and bn fail
2-3 of 8 lines whose audio is right, and a gate that red-flags a quarter of
correct output is not a gate. Only **te (9/9) and kn (8/8)** are listed. The
other three keep the carrier -- it improves their audio either way -- and fall
back to the presence check, which is honest about checking less. The remaining
seven languages have no commercially licensed voice, cannot be dubbed on the
default configuration, and so have no shipping audio to validate a reader
against. `tests/test_short_line_gate.py` asserts exactly which languages are
in which group, so the gap cannot quietly close. Level is checked
alongside it, because wav2vec2 normalises its input and a clip at -50 dBFS
decodes perfectly. Where there is no reader, the line falls back to a
presence check (audible and *pitched*: voiced fraction >= 0.30, against 0.00
for silence, white noise and a click) and the report records that it was not
read.

**Scoring by forced alignment instead of free decoding was measured and
rejected.** The idea: score the per-token Viterbi log-likelihood of the
*expected* text (torchaudio `forced_align`), so the reader only has to
confirm, not transcribe. First, the blank: these Vakyansh models emit `<s>`
(id 0) as the CTC blank, not their configured `<pad>`, and aligning with
`<pad>` scores every clip as garbage -- the blank has to be read off the
model (argmax on silence), and the measurement checked that before scoring
anything. With that right, on production clips (median of 3 draws; 8 lines
x 6 clips in each of two draw sets per language, each clip also scored
against the 7 other lines as a wrong-word case) it separates correct clips
from wrong-word clips well -- AUC 0.978-0.998 for hi/mr/bn in both sets,
lower 95% bound >= 0.957 -- and false-rejects 4-10% of correct clips at a
threshold calibrated on the other set. By the rule set
before looking, Hindi passed and Marathi and Bengali did not (noise passed
the threshold: bn rejected only 59-60 of 96 silence/noise clips).

It then failed the test that matters. Cut 100-150 ms off the head or tail of
a correct clip and it still passes: Hindi 94-100% of such clips, Bengali
100%. Of the real cut defects in these renders (the word truncated, CTC
reading `তি` for `সত্যিই`, `খুনি` for `এখনই`), it passed 4 of 7 in Marathi
and 12 of 14 in Bengali. A likelihood averaged over tokens is carried by the
tokens that are there; scoring the worst single token instead did not fix
it (Hindi still passed 100% of 150 ms head cuts; Bengali AUC fell to
0.925-0.945). A check that passes a truncated word is not a wrong-word
check, so it is not used anywhere.

**A bigger reader does not rescue them either.** `facebook/mms-1b-all` with
each language's adapter was scored against the current reader on the same
480 production clips (one render each, cut as shipped; 78-80 complete clips
per language per set), by a rule fixed first: false-reject <= 1/8, >= 80% of
150 ms head and tail truncations caught, no real mid-word clip passed, <= 5%
wrong-word acceptance, every silence/noise clip rejected -- in both sets. It
failed the first two for every language, in both sets:

| | hi | mr | bn |
|---|---|---|---|
| false-reject, MMS (sets A / B) | 67.5% / 67.5% | 19.2% / 15.2% | 20.0% / 22.5% |
| false-reject, Vakyansh | 25.0% / 31.2% | 24.4% / 26.6% | 17.5% / 17.5% |
| head / tail truncation caught, MMS | 50-62% / 73% | 49-60% / 28-38% | 47-55% / 44-55% |

Both readers rejected every real mid-word clip, every null, and never
accepted another line's text. (These are single renders; the gate reads the
median of three, which is why its per-line numbers above are better. The
comparison between readers is like for like.) IndicConformer is gated on
Hugging Face and was not evaluated -- no token is available on this machine;
IndicWhisper covers Hindi only and is autoregressive, the property that made
Whisper unusable on short clips in the first place. So hi, mr and bn stay on
the presence check, and the report keeps saying so.

Two Whisper-side alternatives were measured and rejected. `initial_prompt`
biasing leaks: a wrong prompt against a degraded clip produced the prompt's
own words (`ఆగు` -> `తెలియదు`), which would manufacture passes. `avg_logprob`
and `no_speech_prob` do not separate -- a bad clip scored -0.218 against a
good one at -0.272, and a good clip's `no_speech_prob` was 0.65 against a bad
one's 0.38.

Lengthening a render to clear the bar remains not allowed; rendering a short
line in a carrier and excising it (#5) is a different thing, because the clip
that ships is still the line at its natural length.

**Enforced by.**
- `tests/test_run_lifecycle.py`: license filtering, registry refusal,
  create-project 422, capabilities fields.
- `tests/test_tts_time_budget.py`: engines don't fit; the mux tempo is the
  whole speed-up.
- `tests/test_voice_clone_budget.py`: the budget, and that one slow draw is
  retimed rather than refused.
- `tests/test_tts_time_budget.py`: short lines take the median of several
  draws, long lines are rendered once, and the same text always renders
  identically; the carrier cut steps back over a stop closure to keep the
  whole word (a real Bengali render is the fixture) but never walks into the
  carrier, the merge is off for Telugu, and `render_line` passes the flag.
- `tests/test_short_line_gate.py`: the ASR floor and the presence check that
  replaces transcription below it, against the "సరే." clip from the run and
  its carrier-sentence control.
- `tests/test_pipeline_hardening.py`: no pitch in prosody; one voice
  reference per speaker.

---

## 8. A run never hangs silently, and the UI speaks the API's actual contract

**What went wrong.**
- A run that no worker picked up, or whose worker died, stayed "processing"
  forever. The UI spun with no error and no way out, because `/process`
  409'd any retry.
- Separately, the frontend's hand-written types had drifted from the API:
  - a QA report shape that didn't exist
  - regenerate stages the backend rejected
  - `detail` arrays rendered as `[object Object]`
  - a language-gate count recomputed client-side

  Every one of these rendered as `undefined` or a silent no-op, never as a
  test failure.

**The rule.**
- While a pipeline task runs, the worker stamps `projects.heartbeat_at`
  every `HEARTBEAT_INTERVAL_SECONDS` (Celery `task_prerun`/`task_postrun`,
  `app/pipeline/liveness.py`).
- `ProjectRead` reports `stalled`, `stalled_reason` and `last_activity_at`.
  A queued or processing run quiet for `STALL_AFTER_SECONDS` is stalled, and
  the reason names the cause: nothing picked it up, or the worker on stage X
  stopped.
- Mutating endpoints (`/process`, upload, segment PATCH, regenerate) return
  409 while a run is active and not stalled. A stalled or failed run can be
  restarted, and the resumable stages continue from where they stopped.
- `regenerate` marks the project `processing`/`regenerate`, re-muxes the
  export, and marks the project `failed` with the reason if either half
  fails.
- The OpenAPI schema is committed at
  `sur-frontend/src/lib/api-contract/openapi.json`
  (`scripts/export_openapi.py`). The backend test fails when it is stale.
  The frontend contract test validates every request the client sends and
  every response field it reads against it.

**Enforced by.**
- `tests/test_run_lifecycle.py`
- `tests/test_openapi_contract_snapshot.py`
- `sur-frontend/src/lib/api.contract.test.ts`
- `sur-frontend/src/lib/runState.test.ts`
- `sur-frontend/e2e/stall.spec.ts` (browser, against a stack with its
  workers stopped)

---

## 9. A TTS line never stops a batch, and no work is lost or done twice

`app/tts_runtime/` (details and every guarantee's test: `docs/tts-runtime.md`).

- Each language has a chain of models (`tts_chains.yaml`). A model that is
  unprovisioned, unhealthy, breaker-open, or -- SYSPIN only -- meets a word in
  `known_bad/<lang>.txt` is skipped with a logged reason; one that crashes,
  hangs, runs out of memory or fails a trigger hands the line to the next.
  When nothing passes, the best attempt is kept and the line is FLAGGED. No
  line raises out of the loop.
- Only allowlisted, commercially usable weights load (`licenses.py`); the
  worker re-checks before loading anything. MMS-TTS, original F5-TTS and XTTS
  are refused by name. Readers are allowlisted too: the Vakyansh models have
  no licence field on Hugging Face and are allowed on their upstream MIT
  licence, which is written down next to them.
- Every model runs in its own process. State is SQLite in WAL mode with
  synchronous=FULL; each line transition commits. Outputs are temp + fsync +
  atomic rename. After kill -9, `resume` requeues the in-flight line,
  re-hashes every finished output, and renders only what is missing.
- Rendering needs no network: workers run with `HF_HUB_OFFLINE` and, in the
  tests, with their sockets blocked. Downloads resume from partial files
  by HTTP Range and are verified against pins taken from Hugging Face's
  public tree API (which works for gated repos without a token).
- Every switch, retry, pause, resume, breaker event, OOM and flag is an
  event and appears in the run report. Nothing falls back silently.
- Thresholds come from measurement, never by hand: `calibration.json` is
  derived by `tts benchmark` on draw set A and evaluated on held-out set B.

**Enforced by.** `tests/test_tts_runtime_units.py`,
`tests/test_tts_runtime_failover.py`, `tests/test_tts_runtime_chaos.py`
(network cut mid-download and mid-batch, kill -9, supervisor, signals, disk
low/full, corrupted model file), `tests/test_tts_runtime_real.py` (SYSPIN
byte-identical to production; offline rendering with sockets blocked).
Every check has a mutation test: `python scripts/mutate_tts_runtime.py`
breaks each one and confirms the named test fails -- 40 of 40 caught.
