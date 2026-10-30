# TTS benchmark

Regenerate with `python -m app.tts_runtime benchmark` (writes
`app/tts_runtime/calibration.json` and `.tts_runtime/bench/`). Test set: the
41 short lines the pipeline's translator produced, the known-bad words, and
20 machine-translated sentences per language; draws 0,1,2 (set A) and 3,4,5
(set B). Draw 0 is exactly what production renders.

Run 2026-09-23 09:49 on Windows AMD64, CPU only.

## Harness verification

- self_cer_zero: 141
- known_perfect: 'అవును.' -> 'అవును', CER 0
- null_samples_rejected: 2

## Trigger calibration (derived on set A, evaluated on held-out set B)

False positives on set B: 3/423 (95% CI [0.002, 0.021])

| corruption | caught | 95% CI |
|---|---|---|
| slowed_x2 | 118/141 | [0.767, 0.889] |
| sped_x2 | 87/141 | [0.535, 0.693] |
| truncated_40pct | 136/141 | [0.920, 0.985] |
| noise_10s | 141/141 | [0.973, 1.000] |
| lead_silence_2_5s | 141/141 | [0.973, 1.000] |
| clipped | 141/141 | [0.973, 1.000] |

## Results

Latency is compute time: 3 render(s) whose wall time spanned a machine sleep are excluded from latency (their audio and CER are kept).

| model | lang | kind | set | lines x seeds | sanity pass | reader CER mean [95% CI] | reader pass | latency s | RTF | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|
| indic_parler | bn | - | - | - | not runnable: ai4bharat/indic-parler-tts: 8 of 8 file(s) not provisioned, e.g. config.json (ga | | | | | |
| indic_parler | hi | - | - | - | not runnable: ai4bharat/indic-parler-tts: 8 of 8 file(s) not provisioned, e.g. config.json (ga | | | | | |
| indic_parler | kn | - | - | - | not runnable: ai4bharat/indic-parler-tts: 8 of 8 file(s) not provisioned, e.g. config.json (ga | | | | | |
| indic_parler | mr | - | - | - | not runnable: ai4bharat/indic-parler-tts: 8 of 8 file(s) not provisioned, e.g. config.json (ga | | | | | |
| indic_parler | te | - | - | - | not runnable: ai4bharat/indic-parler-tts: 8 of 8 file(s) not provisioned, e.g. config.json (ga | | | | | |
| indicf5 | bn | - | - | - | not runnable: ai4bharat/IndicF5: 6 of 6 file(s) not provisioned, e.g. checkpoints/vocab.txt (g | | | | | |
| indicf5 | hi | - | - | - | not runnable: ai4bharat/IndicF5: 6 of 6 file(s) not provisioned, e.g. checkpoints/vocab.txt (g | | | | | |
| indicf5 | kn | - | - | - | not runnable: ai4bharat/IndicF5: 6 of 6 file(s) not provisioned, e.g. checkpoints/vocab.txt (g | | | | | |
| indicf5 | mr | - | - | - | not runnable: ai4bharat/IndicF5: 6 of 6 file(s) not provisioned, e.g. checkpoints/vocab.txt (g | | | | | |
| indicf5 | te | - | - | - | not runnable: ai4bharat/IndicF5: 6 of 6 file(s) not provisioned, e.g. checkpoints/vocab.txt (g | | | | | |
| syspin | bn | short | A | 8x3 | 23/23 | - - | - | 13.856 | 25.414 | 1870.300 |
| syspin | bn | short | B | 8x3 | 22/24 | - - | - | 12.235 | 22.602 | 1907.600 |
| syspin | bn | sentence | A | 20x3 | 60/60 | 0.111 [0.075, 0.156] | 55/60 | 6.529 | 2.062 | 2053.800 |
| syspin | bn | sentence | B | 20x3 | 60/60 | 0.100 [0.068, 0.136] | 59/60 | 6.783 | 2.114 | 2053.800 |
| syspin | hi | short | A | 8x3 | 24/24 | - - | - | 12.086 | 20.682 | 2053.800 |
| syspin | hi | short | B | 8x3 | 24/24 | - - | - | 11.623 | 20.415 | 2053.800 |
| syspin | hi | sentence | A | 20x3 | 60/60 | 0.035 [0.022, 0.049] | 60/60 | 6.325 | 1.649 | 2053.800 |
| syspin | hi | sentence | B | 20x3 | 60/60 | 0.031 [0.018, 0.048] | 60/60 | 6.105 | 1.599 | 2053.800 |
| syspin | kn | short | A | 8x3 | 24/24 | 0.009 [0.000, 0.028] | 24/24 | 11.295 | 19.363 | 2642.200 |
| syspin | kn | short | B | 8x3 | 24/24 | 0.042 [0.000, 0.125] | 24/24 | 10.559 | 18.897 | 2642.200 |
| syspin | kn | sentence | A | 20x3 | 60/60 | 0.021 [0.011, 0.032] | 60/60 | 7.175 | 2.047 | 2642.200 |
| syspin | kn | sentence | B | 20x3 | 59/60 | 0.022 [0.011, 0.034] | 60/60 | 6.999 | 2.013 | 2642.200 |
| syspin | mr | short | A | 8x3 | 24/24 | - - | - | 14.129 | 27.664 | 2642.200 |
| syspin | mr | short | B | 8x3 | 24/24 | - - | - | 12.630 | 25.074 | 2642.200 |
| syspin | mr | sentence | A | 20x3 | 60/60 | 0.109 [0.086, 0.132] | 60/60 | 6.985 | 2.057 | 2642.200 |
| syspin | mr | sentence | B | 20x3 | 60/60 | 0.101 [0.077, 0.129] | 59/60 | 7.026 | 2.091 | 2642.200 |
| syspin | te | short | A | 9x3 | 27/27 | 0.033 [0.000, 0.088] | 26/27 | 12.509 | 20.439 | 2642.200 |
| syspin | te | short | B | 9x3 | 27/27 | 0.069 [0.019, 0.124] | 26/27 | 12.338 | 19.790 | 2642.200 |
| syspin | te | sentence | A | 20x3 | 60/60 | 0.036 [0.021, 0.056] | 59/60 | 6.613 | 2.105 | 2642.200 |
| syspin | te | sentence | B | 20x3 | 60/60 | 0.041 [0.026, 0.058] | 60/60 | 6.724 | 2.155 | 2642.200 |

## Chain order

- **bn**: syspin > indic_parler > indicf5 -- only runnable model: syspin; the others are not runnable here (see table) -- order unchanged
- **hi**: syspin > indic_parler > indicf5 -- only runnable model: syspin; the others are not runnable here (see table) -- order unchanged
- **kn**: syspin > indic_parler > indicf5 -- only runnable model: syspin; the others are not runnable here (see table) -- order unchanged
- **mr**: syspin > indic_parler > indicf5 -- only runnable model: syspin; the others are not runnable here (see table) -- order unchanged
- **te**: syspin > indic_parler > indicf5 -- only runnable model: syspin; the others are not runnable here (see table) -- order unchanged
