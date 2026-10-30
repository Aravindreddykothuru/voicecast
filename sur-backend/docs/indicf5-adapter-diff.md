# IndicF5 adapter vs the real `model.py`

Our adapter (`app/tts_runtime/adapters/indicf5.py`) was written from the
model card, because the repository was gated and its remote code could not
be read. It can now be read: `ai4bharat/IndicF5` at the pinned revision
(`ba85abedf18d…`, see `app/tts_runtime/pins/indicf5.json`) is provisioned and
`model.py` is on disk.

This is the diff. Three of the adapter's guesses were right, and four things
were wrong — one of them the kind that produces confident audio from a model
carrying none of its own weights.

## What the adapter got right

| guess | real `model.py` |
|---|---|
| `model(text, ref_audio_path=…, ref_text=…)` | `def forward(self, text, ref_audio_path, ref_text)` — exact |
| 24 kHz output | `sf.write(buffer, audio, samplerate=24000, …)` |
| int16 → float32 / 32768 | the repo's own `__main__` does precisely this |
| Vocos fetched from the hub at load time | `load_vocoder(vocoder_name="vocos", is_local=False, …)` |

The local-Vocos interception was the right call: `load_vocoder` does reach
for `charactr/vocos-mel-24khz` over the network, and the adapter serves it
from the verified local copy instead.

## 1. The weights can silently not load, and nothing would say so

`model.safetensors` holds **447 tensors**: 364 under `ema_model.` and 83
under `vocoder.`. Every key carries a `_orig_mod.` segment, for example:

```
ema_model._orig_mod.transformer.input_embed.conv_pos_embed.conv1d.0.weight
```

That prefix exists because `INF5Model.__init__` wraps both submodules in
`torch.compile(...)` and the checkpoint was saved from the compiled model.
The keys therefore only line up if `torch.compile` actually wraps the
modules again at load time.

`torch.compile` needs Triton, which is not available on Windows, and this
deployment is Windows on CPU. If the wrap degrades to a no-op the module
tree has no `_orig_mod` level, **none of the 447 tensors match**, and
`from_pretrained` reports them as unexpected and the model's own parameters
as newly initialised — as a warning. It returns a working object that
speaks with untrained weights.

This is CONTRACTS.md #1 exactly, and the adapter walked straight into it by
calling `AutoModel.from_pretrained` directly, which is the one thing that
contract forbids.

**Fixed:** the adapter now reads the checkpoint's key list and checks it
against the loaded module's `state_dict()` after `from_pretrained`. If the
IndicF5 tensors did not land, it raises instead of rendering. A model that
cannot prove it loaded its own weights does not get to speak.

(The three commented-out lines in `model.py` that would download and
`load_state_dict(..., strict=False)` the safetensors by hand are *not* a
bug: `from_pretrained` does that job. They are dead code, and the
`strict=False` in them would have hidden the same problem.)

## 2. `vocab.txt` is fetched from the hub, by repo id, at load time

```python
vocab_path = hf_hub_download(config.name_or_path, filename="checkpoints/vocab.txt")
```

Loading from a local folder makes `config.name_or_path` that folder's path,
so this asks the hub for a repo whose id is a Windows path. With
`HF_HUB_OFFLINE=1` — which the adapter sets, deliberately, so rendering
never depends on the network — it fails outright.

The adapter's local-first shim only covered `charactr/vocos-mel-24khz`.

**Fixed:** the shim now also serves `checkpoints/vocab.txt` (and any other
file the repo ships) from the verified local folder, for both the repo id
and the local path. The file is already provisioned and hash-locked.

## 3. `remove_sil=True` keeps half a second of silence

`config.json` ships `remove_sil: true`, and `forward` then runs

```python
silence.split_on_silence(..., min_silence_len=1000, silence_thresh=-50, keep_silence=500, ...)
```

so output keeps up to **500 ms** of silence at each edge, before a loudness
normalise to −20 dBFS. The runtime's calibrated silence trigger allows
0.30 s (`calibration.json`). Every IndicF5 line would fail that check on
arrival — the same way Indic Parler failed it for five languages in
`docs/tts-warmup-run.json`.

**Fixed:** the adapter passes `remove_sil` and `speed` explicitly from its
options rather than inheriting the shipped defaults, so this is a decision
with a number behind it instead of an accident.

## 4. The `f5_tts` package was never recorded as a dependency

`model.py` imports `f5_tts.infer.utils_infer` and `f5_tts.model.DiT`. The
adapter's docstring listed `transformers<4.50` and `numpy<=1.26.4` for
`.venv-indicf5` but not the package that actually does the synthesis.
`config.json` pins `transformers_version: 4.49.0`, which confirms the first
constraint.

Note for the licence register: the **f5-tts package** is MIT, while the
**SWivid/F5-TTS weights** are CC-BY-NC and are refused by name in
`licenses.FORBIDDEN`. IndicF5 ships its own weights under MIT, so using the
package to run them is fine; what must never happen is loading SWivid's
checkpoint. Recorded in `docs/tts-licences.md`.

**Fixed:** recorded in the adapter docstring and in `requirements-indicf5.txt`.

## Still not done

The smoke test. It needs `.venv-indicf5` with `f5-tts` installed, and the
box is mid-benchmark; `torch.compile` on Windows CPU is also the open
question that decides whether finding 1 is latent or active. IndicF5 stays
**DISABLED** regardless: no reference clip in
`app/tts_runtime/data/tts_references.yaml` has verified consent, which is
the condition `availability()` already enforces.
