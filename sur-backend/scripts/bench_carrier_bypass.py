"""Carrier bypass, question 2: does the alternative render ISOLATED short
words correctly, or does it need the carrier-and-cut scaffold SYSPIN needs?

Answered for Telugu only, and deliberately so: the carrier decision needs a
reader that is reliable on ONE-WORD clips, and only te and kn have one
(CONTRACTS.md #7). mr/bn short-line readers false-reject too often to decide
anything, so this script refuses to pretend otherwise.

Arms, same 9 te short lines, 3 seeds x 2 draw sets:
  - mms isolated   : facebook/mms-tts-tel, no carrier   (the question)
  - syspin+carrier : read from the committed benchmark  (today's production)
SYSPIN's isolated behaviour is already known and is not re-measured here:
8 one-word lines scored 2/8 intelligible alone against 8/8 after a carrier
(CONTRACTS.md #7), which is why the scaffold exists at all.

    python bench_carrier_bypass.py <out_dir>
"""
import hashlib
import json
import os
import pathlib
import sys
import time

BACKEND = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from app.tts_runtime.benchmark import SET_A, SET_B, wilson  # noqa: E402
from app.tts_runtime.checks import READER_MAX_CER, cer  # noqa: E402

OUT, LANG, REPO = sys.argv[1], "te", "facebook/mms-tts-tel"
DATA = os.path.join(BACKEND, "app/tts_runtime/data")
BENCH_ROWS = os.path.join(BACKEND, ".tts_runtime/bench/read.json")


def main() -> int:
    from transformers import AutoTokenizer, VitsModel

    from app.providers.loading import load_hf_model
    from app.tts_runtime.adapters.vakyansh import VakyanshReader

    words = json.load(open(os.path.join(DATA, "benchmark_short_words.json"), encoding="utf-8"))["langs"][LANG]
    reader = VakyanshReader(None, {})
    reader.load("cpu")

    # Harness: the reader must reproduce a score the committed run recorded.
    rows = json.load(open(BENCH_ROWS, encoding="utf-8"))
    import soundfile as sf

    ref = next(r for r in rows if r.get("lang") == LANG and r.get("kind") == "short"
               and r.get("cer") is not None and os.path.exists(r.get("path", "")))
    x, sr = sf.read(ref["path"], dtype="float32")
    got = round(cer(ref["text"], reader.read(x, sr, LANG)), 4)
    if abs(got - ref["cer"]) > 1e-4:
        raise SystemExit(f"HARNESS BROKEN: re-scored {ref['path']} as {got}, committed run said {ref['cer']}")
    print(f"harness verified: reader reproduces committed te short score {got}", flush=True)

    tok = AutoTokenizer.from_pretrained(REPO)
    model = load_hf_model(VitsModel, REPO).eval()   # never bare from_pretrained: CONTRACTS.md #1
    sr_out = model.config.sampling_rate
    import torch

    mms_rows = []
    for text in words:
        for draw in SET_A + SET_B:
            inputs = tok(text, return_tensors="pt")
            torch.manual_seed(int(hashlib.sha256(f"{draw}:{text}".encode()).hexdigest()[:8], 16))
            with torch.inference_mode():
                wav = model(**inputs).waveform[0].float().numpy()
            hyp = reader.read(wav, sr_out, LANG)
            mms_rows.append({"text": text, "draw": draw, "set": "A" if draw in SET_A else "B",
                             "cer": round(cer(text, hyp), 4), "hyp": hyp,
                             "audio_s": round(len(wav) / sr_out, 2)})
        print(f"  {text!r} done", flush=True)

    def arm(rs):
        ok = sum(1 for r in rs if r["cer"] <= READER_MAX_CER)
        lo, hi = wilson(ok, len(rs))
        return {"renders": len(rs), "pass": ok, "pass_rate": round(ok / len(rs), 3),
                "pass_ci95": [round(lo, 3), round(hi, 3)],
                "mean_cer": round(sum(r["cer"] for r in rs) / len(rs), 4)}

    syspin_short = [r for r in rows if r.get("lang") == LANG and r.get("kind") == "short"
                    and r.get("cer") is not None]
    result = {
        "created": time.strftime("%Y-%m-%d %H:%M"), "lang": LANG, "lines": len(words),
        "licence_note": "facebook/mms-tts-* is CC-BY-NC-4.0: benchmark-only, never shippable here.",
        "mms_isolated_no_carrier": {s: arm([r for r in mms_rows if r["set"] == s]) for s in ("A", "B")},
        "syspin_with_carrier_committed": arm(syspin_short),
        "bar": f"a render passes at CER <= {READER_MAX_CER}",
    }
    both = all(result["mms_isolated_no_carrier"][s]["pass_ci95"][0] >= 0.8 for s in ("A", "B"))
    result["decision"] = (
        "bypass the carrier for mms on te: isolated short words pass with the 95% CI lower bound >= 0.8 in both sets"
        if both else
        "keep the carrier for mms: isolated short words do not clear a 0.8 lower bound in both sets")
    result["does_not_transfer"] = ("Says nothing about indic_parler or indicf5, which were never run; "
                                   "and nothing about mr/bn, whose short-line readers are unreliable.")
    json.dump(mms_rows, open(os.path.join(OUT, "carrier_bypass_rows.json"), "w", encoding="utf-8"), ensure_ascii=False)
    path = os.path.join(OUT, "bench_carrier_bypass.json")
    json.dump(result, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    print("report:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
