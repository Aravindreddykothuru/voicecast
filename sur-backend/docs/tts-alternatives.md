# Ungated alternatives, measured 2026-09-30

Indic Parler-TTS, IndicF5 and IndicConformer are gated and no `HF_TOKEN` is
reachable on this machine: all three return **401 GatedRepoError** for a bare
`config.json`. So the questions they were meant to answer were put to the one
alternative that is ungated and already has an adapter's worth of prior art
here, `facebook/mms-tts-*`.

**Licence, stated first.** `facebook/mms-tts-*` is **CC-BY-NC-4.0 —
non-commercial**. It can never ship in this product. It was **not** added to
the runtime's allowlist: `licenses.FORBIDDEN` still refuses it by name, so no
chain can select it. These runs render it *outside* the runtime
(`scripts/bench_mms_vs_syspin.py`) and score it with the runtime's own
functions. Benchmark-only, and the scripts say so in their own docstrings.

**Loading.** The weights are loaded through `app/providers/loading.py::load_hf_model`,
never bare `from_pretrained`. That matters: transformers reports ~128 keys
"newly initialized" for these checkpoints, and a partially random model would
have produced a confident, false CER. The guard confirmed every one of those
keys is the benign pairwise `weight_g/weight_v` → `parametrizations.weight.original0/1`
rename (CONTRACTS.md #1), so the model is fully loaded.

## Question 1 — does an alternative beat SYSPIN on mr/bn?

Protocol identical to `docs/tts-benchmark.md`: the same 20 sentences per
language, the same draw sets (A = seeds 0,1,2; B = 3,4,5) seeded per
`(draw, text)`, the same Vakyansh reader at the same revision, the same CER,
the same bootstrap CI over per-line means. The SYSPIN arm is not re-rendered
— it is read from the committed run. Harness verified first: the reader
reproduced a committed score exactly (mr, CER 0.0588) and silence scored ~1.

Mean sentence CER, 20 lines x 3 seeds per set (lower is better):

| lang | set | MMS-TTS | 95% CI | SYSPIN | 95% CI | verdict |
|---|---|---|---|---|---|---|
| mr | A | 0.1660 | [0.1371, 0.1950] | **0.1085** | [0.0862, 0.1324] | SYSPIN's CI entirely below |
| mr | B | 0.1601 | [0.1339, 0.1881] | **0.1014** | [0.0769, 0.1293] | SYSPIN's CI entirely below |
| bn | A | 0.0891 | [0.0542, 0.1303] | 0.1108 | [0.0753, 0.1560] | CIs overlap |
| bn | B | 0.0963 | [0.0646, 0.1379] | 0.0999 | [0.0677, 0.1361] | CIs overlap |

**No chain changed.** The pre-registered rule is unchanged and was applied as
written: a model moves ahead only if its sentence-CER 95% CI lies entirely
below the other's in **both** draw sets.

- **mr: SYSPIN is significantly better.** MMS is worse by ~0.05 CER and
  SYSPIN's CI sits entirely below MMS's in both sets.
- **bn: MMS looks nominally better in both sets** (0.089 vs 0.111, 0.096 vs
  0.100) **and that is not enough.** The CIs overlap in both sets, so the
  rule is not met and the order stands. This is exactly the case the rule
  exists for; it was not relaxed because the numbers leaned the right way.
  It is also moot for shipping: CC-BY-NC.

## Question 2 — do isolated short words need the carrier?

Answered for **te only**, deliberately: deciding this needs a reader that is
reliable on one-word clips, and only te and kn have one. mr/bn short-line
readers false-reject too often to decide anything.

9 te short lines x 3 seeds per set, pass = reader CER <= 0.35:

| arm | set A | set B | mean CER |
|---|---|---|---|
| MMS, isolated, no carrier | 12/27 (44%) CI [0.28, 0.63] | 10/27 (37%) CI [0.22, 0.56] | 0.62 |
| SYSPIN, with carrier (committed run) | 52/54 (96%) CI [0.88, 0.99] | | 0.05 |

**Keep the carrier.** MMS fails isolated short words about as badly as SYSPIN
does — the scaffold is not a SYSPIN quirk but a property of these small VITS
voices. No bypass for any model. This says nothing about Indic Parler or
IndicF5, which have still never run.

## Reproduce

```
python scripts/bench_mms_vs_syspin.py <out_dir> mr,bn     # docs/tts-alternatives-mms.json
python scripts/bench_carrier_bypass.py <out_dir>          # docs/tts-alternatives-carrier-bypass.json
```
