# TTS runtime: failover chains, crash-safe jobs, offline rendering

`app/tts_runtime/` renders lines through a chain of models per language and
keeps going when a model, the network, the disk or the process fails. It is
a library plus a `tts` command. The pipeline's TTS stage renders through it
when `TTS_USE_RUNTIME=true` (see "Wired into the pipeline" below).

```
make setup-tts                     # venvs, verified downloads, warm-up (idempotent)
python -m app.tts_runtime run spec.json        # or: tts run spec.json
python -m app.tts_runtime resume <job_id>      # after anything at all
python -m app.tts_runtime supervise spec.json  # auto-restart + auto-resume until DONE
python -m app.tts_runtime status | report <job_id> | warmup | benchmark
```

A spec: `{"job_id"?, "out_dir"?, "lines": [{"text", "lang", "speaker"?, "emotion"?, "scene"?}]}`.
Exit codes: 0 DONE, 3 interrupted (resume), 1 error, 2 benchmark harness broken.

## Models and licences

| model | licence | weights | status on the dev box |
|---|---|---|---|
| `syspin` | CC-BY-4.0 | pinned + sha256 (`providers/tts/syspin_manifest.py`) | runs |
| `indic_parler` | Apache-2.0 | gated; pinned by git id (`pins/indic_parler.json`) | **not runnable: no HF_TOKEN** |
| `indicf5` | MIT | gated; pinned (`pins/indicf5.json`, `pins/vocos.json`) | **not runnable: no HF_TOKEN, no verified reference clip** |
| `vakyansh` (reader) | MIT (upstream repo; HF cards have no licence field) | HF cache at pinned revisions | runs |
| `parler_tiny` (tests only) | Apache-2.0 | `pins/parler_tiny.json` | runs (English only) |

`licenses.py` is the allowlist: MMS-TTS, original F5-TTS and XTTS are refused
by name with the reason; anything unlisted is refused; test-only models
cannot join a chain. Checked when the config loads and again inside every
worker before it loads anything.

**Pinning gated repos without a token.** Hugging Face's tree API is public
even for gated repos: it gives each file's size and git object id and masks
only the LFS sha256. A small file's git id is `sha1("blob <n>\0"+content)`;
an LFS file's is the sha1 of its pointer, which embeds its sha256. So
hashing a download and rebuilding the pointer verifies it exactly. Proven on
the ungated parler-tts-tiny-v1, where the sha256 is shown: 3 of 3 files
match (`tests/test_tts_runtime_units.py`). After verification each file's
sha256 goes into a `VERIFIED.json` lock; workers re-hash before loading.

## How a line is rendered

Walk the language's chain (`tts_chains.yaml`). Skip, with a logged reason:
a model that does not speak the language, is not provisioned, failed its
health check, has an open circuit breaker, or -- SYSPIN only -- meets a word
in `known_bad/<lang>.txt`. Otherwise synthesize in the model's own worker
process (retry once with a fresh draw), then run the triggers on the raw
output: empty/NaN, speech rate outside the calibrated range, lead/trail
silence, clipping, reader CER > 0.35 (te/kn), readers disagreeing. Pass:
done. Fail: next model. Nothing passes: a SYSPIN render skipped only for a
known-bad word is tried last so the line is never silent, and the best
attempt is kept and FLAGGED. A line never stops a batch.

Each model runs in its own process with its own interpreter
(`interpreters:` in the config): parler-tts pins transformers==4.46.1, which
neither backend venv can hold. A crash, hang (killed at `synth_s`) or OOM
kills only that process. OOM: the worker frees its cache, the least
recently used other model is unloaded, one retry, then a restart on CPU if
the model supports it, else the next model. At most `max_models_on_gpu`
models are resident.

## Resilience guarantees and what proves them

| guarantee | mechanism | test |
|---|---|---|
| model 1 crashes/garbage/NaN/empty/clips/silence/exception -> model 2 | worker isolation + triggers | `test_model_1_failing_hands_the_line_to_model_2` (7 modes) |
| hung model killed | request timeout kills the process | `test_a_hung_model_is_killed_at_the_timeout...` |
| all models fail -> flagged, batch completes, reported | best-severity attempt kept | `test_every_model_failing_flags_the_line...` |
| network cut mid-download resumes | `.partial` + HTTP Range, hash, atomic rename | `test_network_cut_mid_download_resumes...`; plus a real 1.27 GB HF download killed at 71 MB and cut by the CDN at 254 MB: resumed both times, sha256 = upstream's |
| offline: network tasks pause, rendering continues, resume by itself | connectivity monitor + PAUSED queue + backoff | `test_network_cut_mid_batch_rendering_continues...` |
| kill -9 mid-render -> resume, no duplicates, no corrupt files | SQLite WAL (synchronous=FULL) state machine, atomic outputs, recovery | `test_kill_9_mid_render_then_resume...` (real CLI subprocess) |
| supervisor auto-restarts | `tts supervise` watchdog | `test_the_supervisor_restarts_a_killed_runner...` |
| SIGINT/SIGTERM/CTRL_BREAK -> finish line, persist, exit 3 | signal handler, workers in their own process group | `test_a_stop_signal_finishes_the_current_line...` |
| GPU OOM ladder | pool | `test_oom_unloads_the_least_recently_used...`, `test_second_oom_falls_back_to_cpu` |
| disk low -> pause + alert, nothing touched | disk guard before every line | `test_low_disk_pauses_the_batch...` |
| ENOSPC mid-write -> no partial file | temp + fsync + rename | `test_atomic_write_on_enospc...`, `test_disk_full_while_writing...` |
| corrupt model file -> caught, never loaded, re-fetched | lock re-hash at load, refetch task | `test_a_corrupted_model_file_is_caught...` |
| DONE output missing/changed -> re-rendered | recovery re-hashes outputs | `test_resume_re_renders_a_done_line...` |
| rendering needs no network | `HF_HUB_OFFLINE`, socket guard in workers | `test_rendering_and_reading_need_no_network`, `test_the_offline_guard_really_blocks_the_network` |
| runtime SYSPIN == production SYSPIN | same `render_line` | `test_syspin_through_the_runtime_is_byte_identical...` |

Every check above has a mutation test: `python scripts/mutate_tts_runtime.py`
breaks each one on purpose and confirms the named test fails, then restores
the file. 43 of 43 caught.

Windows note: `kill -9` is `TerminateProcess` (what the tests use); SIGTERM
cannot be caught there, so graceful stop is Ctrl+C / CTRL_BREAK. On Linux
use SIGTERM, or systemd:

```ini
[Service]
WorkingDirectory=/srv/sur-backend
Environment=HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ExecStart=/srv/sur-backend/.venv/bin/python -m app.tts_runtime resume %i
Restart=on-failure
KillSignal=SIGTERM
TimeoutStopSec=120
```

## Measured on this machine (CPU only, 2026-09-23)

Full table: `docs/tts-benchmark.md` (`tts benchmark`; 846 renders, 3 seeds x
2 draw sets per line).

- **Triggers.** Calibrated on set A, evaluated on held-out set B: 3 of 423
  renders flagged (0.7%, 95% CI [0.2%, 2.1%]). Two of the three are 0.22 s
  Bengali renders of a 5-character word at 22.8 chars/s -- the same failure
  mode a 0.21 s draw of a one-word line showed before (CONTRACTS.md #7), so
  they are plausibly real defects, unverifiable because the Bengali reader is
  not reliable on short lines. The third (a slow Kannada sentence, reader CER
  0.036) is a genuine false alarm. Corruptions caught: 10 s of noise, 2.5 s
  of leading silence and clipping 141/141; truncation to 40% 136/141 (96%);
  slowed x2 118/141 (84%); **sped x2 only 87/141 (62%)** -- the speech-rate
  upper bound is the weakest edge, because SYSPIN's own rate varies enough
  that doubling it often stays inside the range.
- **SYSPIN quality** (reader CER, 20 sentences x 3 seeds per set): kn 0.021,
  hi 0.031-0.035, te 0.036-0.041, mr 0.101-0.109, bn 0.100-0.111. Short
  lines, where a reader is reliable: kn CER 0.009-0.042 (24/24 pass), te
  0.033-0.069 (26/27).
- **Speed.** A sentence takes 6.1-7.2 s (RTF ~1.6-2.2); a short line takes
  10.6-14.1 s, because it is three draws of carrier-plus-word. Peak RSS with
  two voices and a reader resident: ~2.6 GB.
- **That 4x headroom is not 4x under load.** In a full-suite run
  (2026-10-01, 524 tests, 37m48s) the first byte-identity case failed with
  `WorkerTimeout: syspin: synth exceeded 60s` -- one short Hindi line, three
  draws, in a freshly spawned worker, at the end of 500 other tests. The same
  file passes alone (9 of 9, 621 s), and the code was byte-identical to the
  run that scored 516 of 516, so this is contention, not a regression. It is
  still the real failure mode: `timeouts.synth_s` bounds the synth call only
  (loading gets its own 900 s), so on a loaded host a line that normally
  takes 14 s can cross 60 s and the pool kills the worker. Production sees
  that as a flagged line, not a wrong one. Not re-tuned here: one observation
  is not a measurement, and the next step is to time first-synth-in-a-fresh-worker
  under deliberate load before touching the number.
- **parler-tts-tiny-v1** (the ungated stand-in that exercises the Parler
  adapter): 40 s to load, 1.64 GB, ~45 s per ~2 s of audio on this CPU. So
  `timeouts.synth_s: 60` is too small for Indic Parler on a CPU host -- on
  this machine most sentences would be killed as timeouts. Raise it (or run
  Parler on a GPU) before putting it in a chain here.
- **A lid-close during the benchmark** put the machine in Modern Standby for
  69 minutes. Nothing was lost; three renders recorded wall times of 129 s,
  262 s and 3958 s. Workers now also report awake time
  (`QueryUnbiasedInterruptTime` on Windows, monotonic elsewhere) and the
  runtime records a `system_suspended` event; the benchmark excludes such
  renders from latency and says how many.

## Network dependencies, all of them

Rendering: **none** (proven with every worker's sockets blocked).
Setup and background tasks only:

1. huggingface.co and its CDN: pinned model files; SYSPIN voices and readers via `huggingface_hub` (which resumes its own partials).
2. HF_TOKEN for the gated repos (read from the environment, never logged, never in `.env`).
3. PyPI, download.pytorch.org and github.com (parler-tts, descript-audiotools) for `.venv-parler`.
4. The connectivity probe (`HEAD https://huggingface.co`, `offline_mode: auto` only).
5. IndicF5's public source downloads its Vocos vocoder at load with no pinned revision; the adapter serves it from the verified local copy. Its gated `model.py` is unread, so this is unverified.

## Wired into the pipeline (feature-flagged, off by default)

`TTS_USE_RUNTIME=true` makes the Celery synthesize stage render SYSPIN in an
isolated subprocess instead of in-process:
`app/providers/tts/runtime_syspin_provider.py` subclasses
`SyspinTTSProvider` and overrides one method, `_render_raw`, to call
`ModelPool.synth("syspin", ..., draw=0)`. Everything else -- prosody rate,
time-fit, voice cloning, loudness normalize -- is the inherited code,
unchanged. **One-line rollback:** set `TTS_USE_RUNTIME=false` (the default)
and restart the worker.

What it buys: a crash, a hang, or a runaway TorchScript call in the render
kills only the worker subprocess. The pool applies SYSPIN's own timeout
(`timeouts.per_model`, 60 s) and the OOM ladder, and every
`app.tts_runtime.pool.WorkerError` is a plain `RuntimeError`, so Celery's
existing `autoretry_for=(Exception,)` retries it with backoff rather than
failing the segment permanently (`_PERMANENT` is `ValueError`/`LookupError`/
`TypeError` only).

What it does **not** change: the audio. Proven byte-for-byte on a fixed
sample set spanning both languages' short lines and sentences, both
genders, with and without emotion -- 5 of 5 sha256 equal
(`tests/test_runtime_syspin_provider.py::test_full_synthesize_output_is_byte_identical`).
This holds because the runtime's `SyspinAdapter` maps `draw=0` to base
offset 0 and then calls the same `common.render_line`, and production only
ever asks for draw 0; a *retry* inside the runtime would use a different
draw, which is why the pipeline path never retries at the pool level.

Measured on a real job (`scripts/verify_wired_tts_byte_identity.py`, project
fc117803, 9 translated Telugu segments, the pipeline's own
`synthesize_segment` with real storage and DB): **9 of 9 lines byte-identical**
between the two paths, with the provider class recorded per pass so the
report cannot claim a path it did not take
(`docs/tts-pipeline-wiring-run.json`). Wall clock was 68.4 s in-process
against 163.1 s through the runtime, but those are not comparable: the
in-process provider had already loaded every voice during its startup
self-check, outside the measured window, while the worker loads lazily
inside it.

The steady-state overhead was then measured on its own
(`scripts/measure_runtime_overhead.py`, `docs/tts-runtime-overhead.json`):
both providers warmed first, then the same 9 lines x 3 runs each,
interleaved so CPU drift hits both arms equally. Per line, `synthesize()`
took **8.204 s** in-process (sd 4.889, n=27) against **8.063 s** through the
runtime (sd 4.496, n=27) -- the runtime is **0.141 s faster per line
(-1.7%)**, which is not a speed-up either: the gap is 3% of the
between-line spread, so what this shows is that IPC plus the WAV round trip
costs nothing measurable next to the render itself. Construction, which
happens once per worker, is the part that differs: 217.4 s in-process (all
voices loaded in the parent's self-check) against 164.4 s for the runtime.
Production viability on this axis: yes, the flag is free per line.

**A whole dub, both ways.** One full 7-stage run per setting, same source
(the 32.56 s two-speaker clip, 9 spoken segments), same target (kn), same
`--asr-check` (`docs/tts-pipeline-flag-e2e.json`). Both passed every check,
and the exported audio is **bit-identical**: one sha256 over 1,041,750
bytes of decoded PCM for both (the digest is abbreviated in the evidence
file because the hook reads a bare 64-char hex string as a secret; the
recomputing command is there instead). Every
scored line matches too -- corpus CER 0.131, per-line 0.042 to 0.282, 0
overruns, 0 overlaps, max tempo 1.3437 in both runs. The worker's own log
line records which provider served each run, so neither can claim a path it
did not take. The synthesize *stage* took 127.8 s against 264.6 s, which is
the lazy first load inside the job, not per-line cost -- and single runs on
a loaded laptop, so no CI is claimed for either number.

Not yet routed through the runtime: the failover chain itself. Production
still renders SYSPIN or fails; it does not fall through to Indic Parler or
IndicF5, because neither is provisioned (both gated) and neither has been
benchmarked against the pre-registered rule. Turning that on is a config
change (`chains:`) plus a benchmark, not a code change.

## Running without a token, stated in config

`tts setup` against the default `tts_chains.yaml` exits non-zero on a
token-less box, naming `indic_parler` and `indicf5` as unprovisioned. That
is deliberate: a chain naming a model nobody downloaded is a half-set-up
deployment, and skipping it quietly is how a language ends up rendering
nothing. The supported way to run anyway is to say so in configuration:

```
make setup-tts TTS_CONFIG=tts_chains.syspin-only.yaml
```

`tts_chains.syspin-only.yaml` declares SYSPIN-only chains for the five
languages SYSPIN speaks and omits the rest, so `submit()` refuses a line it
cannot serve instead of queueing it to be flagged forever. There is no
failover in that profile -- which is what the default config already does
in practice today, only now it is a stated choice rather than a shrug.

The one ungated alternative was benchmarked under the same protocol and did
not displace anything: `docs/tts-alternatives.md`. It is also CC-BY-NC, so
`licenses.FORBIDDEN` still refuses it and the runs were done outside the
runtime.
