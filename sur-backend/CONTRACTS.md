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

**Enforced by.** `tests/test_model_loading_is_fail_loud.py`,
`tests/test_startup_checks.py`. The first test is the exact ehcalabres key
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
- TTS availability per language is derived from the voice that speaks it
  (`Language.mms_tts`), never a bare `True`. Every row used to say
  `tts_supported=True` while the configured TTS model (CosyVoice2) could not
  speak any of the twelve languages.
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
