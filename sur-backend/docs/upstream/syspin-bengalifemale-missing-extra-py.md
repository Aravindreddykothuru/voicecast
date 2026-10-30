# DRAFT — upstream report for SYSPIN (not yet posted)

**Where to post:** a Discussion on
<https://huggingface.co/SYSPIN/tts_vits_coquiai_BengaliFemale/discussions>
(Hugging Face repos take reports as Discussions, not Issues).

**Status:** draft for a maintainer of this project to review and post. Nothing
has been sent to SYSPIN.

---

## Title

`tts_vits_coquiai_BengaliFemale` is missing `extra.py`, so the model cannot be
loaded the way every other SYSPIN release is

## Body

Thank you for releasing these voices under CC-BY-4.0 — we use the Telugu,
Hindi, Kannada, Marathi and Bengali models in a dubbing pipeline, with
attribution.

We found that the Bengali female release does not include `extra.py`, which
every other release we use ships beside the TorchScript weights and
`chars.txt`:

| repo | revision | files |
|---|---|---|
| `SYSPIN/tts_vits_coquiai_BengaliFemale` | `4aa58c497f71a19abe906a5845eb66930b40ceb2` | `.gitattributes`, `README.md`, `bn_female_vits_30hrs.pt`, `chars.txt`, `jit.wav`, `jit_infer.py` — **no `extra.py`** |
| `SYSPIN/tts_vits_coquiai_BengaliMale` | `93d425c03945` | `bn_male_vits_30hrs.pt`, `chars.txt`, `extra.py`, `jit_infer.py` |
| Telugu, Hindi, Kannada, Marathi (male and female) | — | each includes `extra.py` |

`extra.py` provides the `VitsConfig` / `CharactersConfig` / `TTSTokenizer`
classes needed to turn text into the token ids the exported graph expects.
Without it the female Bengali voice cannot be tokenized, so loading fails:

```
FileNotFoundError: [Errno 2] No such file or directory:
  .../models--SYSPIN--tts_vits_coquiai_BengaliFemale/snapshots/4aa58c497f71.../extra.py
```

**What we observed about the file itself:** `extra.py` is byte-identical in all
nine releases that include it (sha256
`3ac9a2fd0cadc470…`, verified across Telugu, Hindi, Kannada and Marathi male
and female, and Bengali male). As a workaround we load the Bengali female
weights with that identical `extra.py` taken from another release. The result
reads back well — a Bengali CTC recogniser transcribes it at a median
character error rate of 0.047 on our test sentences, better than the male
voice's 0.116 — so the weights and `chars.txt` appear to be fine. Only the
file is missing.

**Request:** could `extra.py` be added to the BengaliFemale repository? If the
omission is intentional (for example, if this voice needs a different
tokenizer), a note in the model card would help — our workaround assumes the
shared file is correct for it, and we would rather know than assume.

## Reproduce

```python
from huggingface_hub import list_repo_files
print(sorted(list_repo_files("SYSPIN/tts_vits_coquiai_BengaliFemale")))  # no extra.py
print(sorted(list_repo_files("SYSPIN/tts_vits_coquiai_BengaliMale")))    # has extra.py
```

---

## Notes for whoever posts this (delete before posting)

- Our side is tracked as issue #6 (closed with the workaround) and in
  `app/providers/tts/syspin_manifest.py`, where BengaliFemale is pinned with
  `ships_extra_py=False`. If SYSPIN publishes the file, bump that pin to the
  new revision, set `ships_extra_py=True`, add its sha256, and confirm it
  equals `EXTRA_PY_SHA256` — then `tts_voice_warnings` goes empty.
- The CER figures above are from this project's own measurement; if quoting
  them, keep "on our test sentences".
