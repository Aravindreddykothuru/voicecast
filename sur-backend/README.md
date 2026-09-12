# Sur — backend

Backend for **Sur**, an emotion-aware AI dubbing pipeline: upload an
English-language video, get back a dub in a target Indic language (default
Telugu) that preserves the speaker's emotional register and, optionally,
their own voice, timed to the original mouth movements.

This repo is the backend only. The frontend (Next.js/TypeScript) is built
separately against the REST + WebSocket contract described below — see
"Frontend integration" at the end of this file if you're wiring up a UI
(e.g. one generated in Stitch) against this API.

## Architecture, in one paragraph

FastAPI serves REST under `/api` and a WebSocket at `/ws/projects/{id}`.
Nothing GPU-bound ever runs inside a request handler: `POST
/api/projects/{id}/process` enqueues a **Celery chain** of seven tasks —
`extract_audio → chunk_and_diarize → transcribe → detect_emotion →
translate → synthesize → mux_export` — each on its own Redis-backed queue,
so CPU work (extraction, muxing) and GPU work (ASR, diarization, emotion,
translation, TTS) scale independently and a failed stage retries alone.
Every stage writes back to a segment row in Postgres; media lives in
S3-compatible object storage (MinIO locally). Progress fans out over Redis
pub/sub to whoever's listening on that project's WebSocket channel.

Every model call goes through a small **provider interface**
(`ASRProvider`, `DiarizationProvider`, `TranslationProvider`,
`EmotionProvider`, `TTSProvider` — see `app/providers/base.py`). Pipeline
tasks never import a model library directly; they ask
`app/providers/registry.py` for "the configured ASR provider" and get back
either a mock or a real implementation depending on an env var. That's what
makes the next section possible.

## Quickstart — running without a GPU

Every provider defaults to `mock`: deterministic, fast, no model download,
no GPU. This is enough to build/test the entire API and pipeline shape
end-to-end — exactly what you need while a frontend (Stitch, Next.js,
whatever) is being wired up against this backend.

```bash
cp .env.example .env
docker compose up --build
```

This starts Postgres, Redis, MinIO, runs Alembic migrations, and brings up
the API (`:8000`), two worker pools, and Flower (`:5555`) for queue
monitoring. Once it's up:

- API docs: http://localhost:8000/docs (OpenAPI schema — this is what a
  frontend's `openapi-typescript` step should point at)
- MinIO console: http://localhost:9001 (user/pass: `sur_minio` /
  `sur_minio_secret`)
- Flower: http://localhost:5555

A full run — create project, upload, process, watch it dub itself with a
sine-tone "voice" and placeholder translations — works with nothing else
installed. That's intentional: the mock TTS provider renders a real,
playable WAV whose duration tracks input text length, so sync-offset math,
waveform players, etc. all have something real to react to.

## Running with real models

Flip the relevant `*_PROVIDER` env var(s) from `mock` to `real` (see
`.env.example`) and, on a GPU-capable worker, install the heavy deps:

```bash
pip install -r requirements-ml.txt
```

You can mix and match — e.g. real ASR + translation, mock TTS — since each
stage's provider is selected independently. `docker-compose.yml`'s
`worker-gpu` service is where you'd point ASR/diarization/emotion/
translation at a GPU runtime (swap its Dockerfile's base image for an
`nvidia/cuda` one). Real providers import their dependencies lazily; if you
set `*_PROVIDER=real` without installing `requirements-ml.txt`, you get a
clear `ProviderNotInstalledError` at startup, not a silent failure
mid-pipeline.

Some real providers (`IndicTrans2Provider`, `PyannoteDiarizationProvider`)
need a HuggingFace token accepted for gated model weights.

**How TTS works.** `TTS_PROVIDER=real` speaks with **MMS-TTS**
(`facebook/mms-tts-<lang>`, one small VITS checkpoint per target language;
the suffix per language lives in `app/capabilities.py`). CosyVoice2 was the
renderer until it was measured: it has never been trained on an Indic
script, and on CPU it spent 30+ minutes on one 3-second Telugu sentence and
returned 21 seconds of audio that Whisper large-v3 transcribed at CER 1.56
(and detected as Korean). MMS renders the same sentence in ~5s at CER 0.09.
Detected emotion is carried as prosody (speaking rate, energy, VITS
variation), weighted by the classifier's confidence -- MMS has no emotion
conditioning, and the code says so rather than pretending otherwise. There
is deliberately no pitch shift: measured, it cost intelligibility on short
lines (CER 0.21 -> 0.36) for an effect a listener barely notices.

Two more things sit between ASR and TTS because the real run needed them:
numerals are spelled out before translation (`app/pipeline/text_normalize.py`;
"at 9" was being dubbed as "at", since MMS vocabularies have no digits), and
TTS is told the time until the *next spoken line* rather than the segment's
own length, so it never squeezes a line that has a pause after it.

To reproduce an end-to-end run and its checks against a running stack:

```powershell
.\.venv\Scripts\python.exe scripts\e2e_dub.py --video path\to\clip.mp4 --target te --asr-check
.\.venv\Scripts\python.exe scripts\e2e_dub.py --project <id> --asr-check   # verify an existing run
```

**Voice cloning** (`clone_voice`, enabled with `TTS_VOICE_CLONE=true`) keeps
CosyVoice2, for the part it can do: *voice conversion*. It re-voices the MMS
speech as the original speaker. The reference is built once per speaker from
their longest lines (~12s) and stored on the speaker row, so every line a
speaker says converts toward the same voice -- per-segment slices (0.5s for a
one-word line) measurably hurt intelligibility. VC has no text model, so the
language limitation doesn't apply. It is roughly 10x slower than real time on
CPU. `/api/capabilities` publishes `voice_clone_available`, and
`/process` refuses `clone_voice` when it's off.

**TTS runs in its own worker, deliberately.** CosyVoice2 is installed from
source and pins `torch==2.3.1`; every other real provider above pins
`torch==2.4.1`. The two cannot share a venv or a Docker image without one
silently breaking the other (this is not theoretical -- it's why the split
exists). MMS-TTS runs in the TTS venv too. Locally, install
`requirements-ml.txt` into `.venv` and `requirements-tts.txt` + CosyVoice
into a second, separate `.venv-tts`, and run the TTS worker from that
interpreter consuming only `q.synthesize`:

```bash
celery -A app.celery_app worker -Q q.synthesize --loglevel=INFO -n tts@%h
```

In Docker, this is `Dockerfile.worker-tts` and the `worker-tts` compose
service (see `docker-compose.yml`) -- it clones CosyVoice at build time,
builds Matcha-TTS's Cython extension, and consumes only `q.synthesize`,
while `worker-gpu` handles every other GPU-bound stage. Model weights are
not baked into either image; download them once into the `models-data`
volume with `.tools/download_cosyvoice.py` (or `huggingface-cli download
FunAudioLLM/CosyVoice2-0.5B`) before setting `TTS_PROVIDER=real`. See
CONTRACTS.md #1.

**Before shipping voice cloning or any cloned-voice output to real users**,
read the PRD's risk list — consent capture, audio watermarking, and
confirming commercial-use licensing on the TTS models are explicit
open items, not implementation details to skip. **MMS-TTS weights are
CC-BY-NC-4.0 (non-commercial).** A commercial deployment needs a
differently-licensed Indic TTS behind the same `TTSProvider` interface
(e.g. `ai4bharat/indic-parler-tts`, Apache-2.0, gated on the Hub).

## Running natively on Windows (CPU)

Docker runs only the infrastructure; the API and workers run from the two
virtualenvs. Each command below is its own long-running terminal.

```powershell
docker compose -f docker-compose.dev.yml up -d        # Postgres :5435, Redis :6380
.\.venv\Scripts\python.exe -m alembic upgrade head
.\scripts\run-local.ps1 -Role api
.\scripts\run-local.ps1 -Role worker -Offline         # every stage except synthesis
.\scripts\run-local.ps1 -Role tts-worker -Offline     # q.synthesize, from .venv-tts
```

What this needs, and why each piece exists:

- **Own ports.** `docker-compose.dev.yml` uses 5435/6380 because other local
  projects' containers hold 5432-5434/6379, and a shared Redis means shared
  Celery broker databases. `docker-compose.test.yml` is tmpfs and TRUNCATEd
  by pytest -- never put dev data in it.
- **ffmpeg paths.** Set `FFMPEG_BINARY`/`FFPROBE_BINARY` in `.env` when
  ffmpeg is not on PATH (a missing binary is a clear, non-retried error).
- **`-Offline`** sets `HF_HUB_OFFLINE=1`: models load from the local cache
  only. That is also how the gated models (pyannote, IndicTrans2) load
  without `HF_TOKEN` once cached -- online, huggingface_hub re-raises
  `GatedRepoError` even for fully cached files.
- **Solo pool.** Celery's prefork pool does not work on Windows;
  `app/celery_app.py` selects `solo` there. One task at a time per worker,
  which is also right for workers holding several GB of models.
- **CosyVoice import path.** `COSYVOICE_SRC_DIR=.tools/CosyVoice` puts the
  source checkout on `sys.path` from inside the provider.
- **Startup checks** load and self-check every model the worker's queues
  need before it accepts work; the worker log shows each one
  (`/startup] ... loaded and self-checked`). Expect ~1 minute (main) and
  ~2 minutes (TTS, with cloning enabled).

## API surface

| Method & path | Purpose |
|---|---|
| `GET /api/projects` | List the current user's projects (dashboard cards) |
| `POST /api/projects` | Create a project — title, target language(s) |
| `POST /api/projects/{id}/upload` | Issue a presigned PUT URL for the source video |
| `POST /api/projects/{id}/upload/confirm` | Confirm the upload completed |
| `POST /api/projects/{id}/process` | Start the pipeline — flags: `preserve_emotion`, `clone_voice`, `lip_sync_aware`, `tts_model` |
| `GET /api/projects/{id}` | Project status, current stage |
| `GET /api/projects/{id}/segments` | All segments — text, translation, emotion, audio URLs |
| `PATCH /api/segments/{id}` | Edit a segment's translated text or override its emotion label |
| `POST /api/segments/{id}/regenerate` | Re-run translate and/or synthesize for one segment only |
| `GET /api/projects/{id}/export` | Final muxed video + QA report |
| `WS /ws/projects/{id}` | `stage_started` / `stage_progress` / `stage_completed` / `segment_ready` / `error` |

`GET /api/projects` isn't in the original PRD's endpoint table but is
needed by the Dashboard screen ("all projects, status at a glance"); it's
implemented the same way as everything else rather than left as a gap.

Auth is stubbed: every request resolves to a `User` row via an
`X-User-Email` header (falling back to `DEV_DEFAULT_USER_EMAIL`), auto-
creating the user on first sight. Swap `app/core/security.py`'s
`get_current_user` for real auth (JWT/OAuth) later — every route already
depends on it, so nothing else changes.

## Data model

`users`, `projects` (target languages, status, feature flags),
`source_videos`, `speakers` (embedding ref, reference clip, consent flag),
`segments` (timing, source/translated text, emotion label + score,
tts_audio_url, sync_offset_pct, status), `export_jobs` (output URL, QA
report JSON). See `app/models/` and `alembic/versions/0001_initial_schema.py`.

**MVP scope note**: a project processes exactly one source video (the most
recently uploaded one) and one target language (`target_languages[0]`).
The schema supports more (both are JSON/array-shaped) — extending
`/process` and the pipeline tasks to fan out over multiple target languages
is Phase 4 ("multi-language batch export") territory, not a redesign.

## Project layout

```
app/
  main.py              FastAPI app, routers, CORS
  config.py             .env-driven Settings (the ONLY place reading os.environ)
  celery_app.py         Celery app + per-stage queue routing
  db.py                 SQLAlchemy engine/session
  logging_conf.py        structured JSON logs, project_id/segment_id bound automatically
  models/                SQLAlchemy models (one file per table)
  schemas/                Pydantic request/response models
  api/                    FastAPI routers + the WebSocket gateway
  storage/                S3/MinIO-compatible object storage interface
  providers/              ASR/diarization/translation/emotion/TTS interfaces
    base.py                the four ABCs pipeline tasks call through
    registry.py             env-var-driven factory (mock vs. real)
    */mock_provider.py       deterministic, GPU-free defaults
    */<real>_provider.py     faster-whisper / IndicTrans2 / MMS-TTS / wav2vec2 / pyannote
    tts/cosyvoice_vc.py      CosyVoice2 voice conversion (clone_voice)
  pipeline/
    tasks.py               the 7 Celery stage tasks
    chain.py                builds the full-project chain (/process)
    regenerate.py            single-segment re-render (/regenerate), split across two queues
    timeline.py              pure: diarization-turn normalization + clip fit planning
    events.py                Redis-pub/sub progress events -> WebSocket
    ffmpeg_utils.py           the only module that shells out to ffmpeg/ffprobe
scripts/run-local.ps1     native Windows launcher for api / worker / tts-worker
alembic/                  migrations
tests/                    pytest suite (SQLite + Celery eager mode + mock providers -- no infra needed)
docker-compose.yml
Dockerfile.api / Dockerfile.worker / Dockerfile.worker-tts
requirements.txt / requirements-ml.txt / requirements-tts.txt
```

## Testing

```bash
pip install -r requirements-dev.txt
docker compose -f docker-compose.test.yml up -d          # Postgres on :5434
export TEST_DATABASE_URL=postgresql+psycopg2://postgres:test@localhost:5434/test  # pragma: allowlist secret
pytest -v --cov=app
```

The suite runs against **real Postgres**, and `TEST_DATABASE_URL` is
required with no default -- a `sqlite://` value is a hard error, and so is
a URL pointing at a production host, because the fixtures reset state with
`TRUNCATE ... CASCADE` between tests.

That requirement is not incidental. The suite used to run on SQLite in
memory, which does not enforce column widths, foreign keys, NOT NULL,
unique constraints or CHECK constraints. `projects.status` shipped as
`VARCHAR(10)` -- wide enough for `"processing"` -- and 137 tests stayed
green until a real project tried to store `"awaiting_language_confirmation"`
and Postgres rejected it in production. Anything that reintroduces a
SQLite fallback reopens that class of bug.

Everything else still needs no GPU and no network:
`celery_app.conf.task_always_eager = True` replaces a real broker/worker,
and an in-memory storage double replaces MinIO/S3. Coverage includes each
provider's mock implementation, the registry's mock/real selection, the
full seven-stage pipeline end-to-end, idempotency of `chunk_and_diarize`
under a retry, the API's project/segment endpoints, cross-user
authorization on every id-bearing route, and a migration-drift check that
fails if `app/models/` and `alembic/versions/` disagree.

## Secret scanning — run this once per clone

`.pre-commit-config.yaml` runs detect-secrets on every commit, but git
hooks live in `.git/hooks`, which **is not cloned**. Until you run:

```bash
pip install pre-commit && pre-commit install
```

a fresh clone has no scanning at all, and a commit containing an AWS key
succeeds silently. Verify it is active by checking that
`.git/hooks/pre-commit` exists.

## Migrations

```bash
make migrate                       # apply
make revision m="add foo column"   # generate a new one after changing app/models/*
```

## Non-functional notes

- **Idempotent, retryable stages**: every task re-derives what work is left
  from the DB (e.g. `chunk_and_diarize` no-ops if segments already exist
  for that video) rather than assuming a clean slate, and Celery is
  configured with `task_acks_late` + `task_reject_on_worker_lost` so a
  killed worker's in-flight task is redelivered, not lost.
- **Structured logging**: every log line carries `project_id`/`segment_id`
  when set (`app/logging_conf.bind_context`), so `docker compose logs |
  grep <project_id>` gets you a whole run.
- **Segment-level caching**: `/regenerate` re-renders exactly one segment
  (translate and/or synthesize, your choice) — this is what keeps
  iteration cheap on the expensive stages, per the PRD's risk list.

## Frontend integration

Point your frontend's `API_BASE_URL` at `:8000` and `WS_BASE_URL` at the
same host for `/ws/projects/{id}`. Generate types from the live OpenAPI
schema rather than hand-writing them:

```bash
npx openapi-typescript http://localhost:8000/openapi.json -o types/sur-api.ts
```

CORS is open to `API_CORS_ORIGINS` (`.env`, defaults to
`http://localhost:3000`) — add your dev origin there if it differs.

## Rotating the Hugging Face token

Two providers download **gated** weights and need a Hugging Face token:
`DIARIZATION_PROVIDER=real` (pyannote) and `TRANSLATION_PROVIDER=real`
(IndicTrans2). Everything else uses open weights and needs nothing.

The token is **never stored in `.env`**. It is read from the process
environment at call time (see `CONTRACTS.md` invariant #4), so rotating it is
a restart, not a code change.

### First-time setup

1. Create a token at <https://huggingface.co/settings/tokens> (read scope).
2. With that same account, accept the licence on each gated model page:
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
   - <https://huggingface.co/ai4bharat/indictrans2-en-indic-1B>
   Access is auto-approved; the token alone is not enough without this step.
3. Export it in the shell that starts the API and each worker:

   ```powershell
   # PowerShell
   $env:HF_TOKEN = "hf_..."
   ```
   ```bash
   # bash
   export HF_TOKEN=hf_...
   ```

   In Docker, pass it through rather than baking it into the image:
   ```yaml
   environment:
     HF_TOKEN: ${HF_TOKEN:?HF_TOKEN must be set}
   ```

`HUGGING_FACE_HUB_TOKEN` is accepted as an alias.

### Rotating

1. Create the new token on Hugging Face; confirm the licences above are
   accepted by that account.
2. Update the value wherever it is injected (CI/CD secret store, systemd unit,
   compose `.env` **outside** the repo, or your shell profile).
3. Restart the API and **every** worker. Nothing caches the token, so a
   restart is sufficient and no code change is needed.
4. Revoke the old token on Hugging Face.
5. Verify: a worker refuses to start with a clear message if the token is
   missing, so a successful boot with `DIARIZATION_PROVIDER=real` confirms the
   new token works.

### If a token leaks

Revoke it first at <https://huggingface.co/settings/tokens>, then rotate as
above. `tests/test_secrets.py` fails the build if a token literal is
committed anywhere in the backend tree, but revocation is still the first
action — a leaked credential is compromised the moment it is shared.
