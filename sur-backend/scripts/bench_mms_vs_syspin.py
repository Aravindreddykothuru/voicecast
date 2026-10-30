"""PATH B: does an ungated alternative beat SYSPIN on mr/bn sentence CER?

facebook/mms-tts-* is CC-BY-NC-4.0 -- NON-COMMERCIAL, so it can never ship in
this product. It is benchmarked here only to answer the question the gated
models were meant to answer, and it is deliberately NOT added to the runtime's
licence allowlist: `licenses.FORBIDDEN` still refuses it by name, so nothing
in a chain can ever select it. This script renders it outside the runtime and
scores it with the runtime's own functions.

Protocol, identical to docs/tts-benchmark.md so the arms are comparable:
  - the same 20 sentences per language (data/benchmark_sentences.json)
  - the same draw sets, A = 0,1,2 and B = 3,4,5, seeded per (draw, text)
  - the same reader (Vakyansh, same revision), the same CER, the same
    bootstrap CI over per-line means
  - the SYSPIN arm is NOT re-rendered: its rows are read from the committed
    benchmark run (.tts_runtime/bench/read.json)

Pre-registered rule, unchanged: a model moves ahead of another only if its
sentence-CER 95% CI lies entirely below the other's in BOTH draw sets.

    python bench_mms_vs_syspin.py <out_dir> [langs]        # default: mr,bn
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

from app.tts_runtime.benchmark import SET_A, SET_B, boot_mean_ci  # noqa: E402
from app.tts_runtime.checks import cer  # noqa: E402

OUT = sys.argv[1]
LANGS = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["mr", "bn"])
MMS_REPO = {"mr": "facebook/mms-tts-mar", "bn": "facebook/mms-tts-ben",
            "hi": "facebook/mms-tts-hin", "te": "facebook/mms-tts-tel"}
DATA = os.path.join(BACKEND, "app/tts_runtime/data")
BENCH_ROWS = os.path.join(BACKEND, ".tts_runtime/bench/read.json")


def sentences(lang):
    return json.load(open(os.path.join(DATA, "benchmark_sentences.json"), encoding="utf-8"))["langs"][lang]


class Reader:
    """Vakyansh, in-process, at the same pinned revision the runtime uses."""

    def __init__(self):
        from app.tts_runtime.adapters.vakyansh import VakyanshReader

        self.r = VakyanshReader(None, {})
        self.r.load("cpu")

    def read(self, wav, sr, lang):
        return self.r.read(wav, sr, lang)


class MMS:
    def __init__(self, lang):
        from transformers import AutoTokenizer, VitsModel

        from app.providers.loading import load_hf_model

        repo = MMS_REPO[lang]
        self.tok = AutoTokenizer.from_pretrained(repo)
        # NOT VitsModel.from_pretrained: that silently random-initialises any
        # key it cannot fill, and a partially random model would produce a
        # confident, false CER. load_hf_model refuses missing weights unless
        # they are the known-benign weight_norm rename, applied pairwise
        # (CONTRACTS.md #1 -- MMS-TTS reports that rename on ~128 keys).
        self.model = load_hf_model(VitsModel, repo).eval()
        self.sr = self.model.config.sampling_rate

    def render(self, text, draw):
        import torch

        inputs = self.tok(text, return_tensors="pt")
        # Seeded per (draw, text), mirroring SyspinVoice.render, so the draw
        # sets vary the same way on both arms.
        torch.manual_seed(int(hashlib.sha256(f"{draw}:{text}".encode()).hexdigest()[:8], 16))
        with torch.inference_mode():
            return self.model(**inputs).waveform[0].float().numpy(), self.sr


def verify_harness(reader, rows):
    """Two broken harnesses have produced false conclusions here before."""
    checks = {}
    texts = [t for lang in LANGS for t in sentences(lang)]
    bad = [t for t in texts if cer(t, t) != 0]
    if bad:
        raise SystemExit(f"HARNESS BROKEN: cer(t,t) != 0 for {bad[:2]}")
    checks["self_cer_zero"] = len(texts)

    # The reader path must reproduce a score the committed run already
    # recorded, or this script's reader is not the benchmark's reader.
    import soundfile as sf

    ref = next((r for lang in LANGS for r in rows
                if r.get("lang") == lang and r.get("kind") == "sentence"
                and r.get("cer") is not None and os.path.exists(r.get("path", ""))), None)
    if ref is None:
        raise SystemExit("HARNESS BROKEN: no committed SYSPIN render to re-score")
    x, sr = sf.read(ref["path"], dtype="float32")
    got = round(cer(ref["text"], reader.read(x, sr, ref["lang"])), 4)
    if abs(got - ref["cer"]) > 1e-4:
        raise SystemExit(f"HARNESS BROKEN: re-scoring {ref['path']} gave CER {got}, "
                         f"the committed run recorded {ref['cer']}")
    checks["reader_reproduces_committed_score"] = {"lang": ref["lang"], "cer": got}

    silence = __import__("numpy").zeros(16000, dtype="float32")
    if cer(ref["text"], reader.read(silence, 16000, ref["lang"])) < 0.9:
        raise SystemExit("HARNESS BROKEN: silence scored well")
    checks["silence_rejected"] = True
    return checks


def per_line_means(rows, lang, draw_set):
    """Mean CER per line over the seeds in one draw set."""
    by_line = {}
    for r in rows:
        if r.get("lang") == lang and r.get("kind") == "sentence" and r.get("set") == draw_set \
                and r.get("cer") is not None:
            by_line.setdefault(r["text"], []).append(r["cer"])
    return [sum(v) / len(v) for v in by_line.values()]


def main() -> int:
    rows = json.load(open(BENCH_ROWS, encoding="utf-8"))
    reader = Reader()
    harness = verify_harness(reader, rows)
    print("harness verified:", json.dumps(harness, ensure_ascii=False), flush=True)

    result = {"created": time.strftime("%Y-%m-%d %H:%M"), "harness": harness,
              "licence_note": "facebook/mms-tts-* is CC-BY-NC-4.0: benchmark-only, never shippable here.",
              "langs": {}}
    mms_rows = []
    for lang in LANGS:
        if lang not in MMS_REPO:
            continue
        t0 = time.time()
        try:
            mms = MMS(lang)
        except Exception as e:
            result["langs"][lang] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"{lang}: MMS unavailable -- {type(e).__name__}", flush=True)
            continue
        for text in sentences(lang):
            for draw in SET_A + SET_B:
                wav, sr = mms.render(text, draw)
                hyp = reader.read(wav, sr, lang)
                mms_rows.append({"lang": lang, "text": text, "draw": draw,
                                 "set": "A" if draw in SET_A else "B",
                                 "kind": "sentence", "cer": round(cer(text, hyp), 4),
                                 "hyp": hyp, "audio_s": round(len(wav) / sr, 2)})
            print(f"  {lang}: {text[:30]!r} done", flush=True)
        arm = {"render_seconds": round(time.time() - t0, 1), "sets": {}}
        for s in ("A", "B"):
            mms_means = [sum(v) / len(v) for v in
                         {t: [r["cer"] for r in mms_rows if r["lang"] == lang and r["set"] == s and r["text"] == t]
                          for t in sentences(lang)}.values()]
            sys_means = per_line_means(rows, lang, s)
            m_mean, m_lo, m_hi = boot_mean_ci(mms_means)
            s_mean, s_lo, s_hi = boot_mean_ci(sys_means)
            arm["sets"][s] = {
                "mms": {"n_lines": len(mms_means), "mean": round(m_mean, 4), "ci95": [round(m_lo, 4), round(m_hi, 4)]},
                "syspin": {"n_lines": len(sys_means), "mean": round(s_mean, 4), "ci95": [round(s_lo, 4), round(s_hi, 4)]},
                "mms_ci_entirely_below_syspin": bool(m_hi < s_lo),
                "syspin_ci_entirely_below_mms": bool(s_hi < m_lo),
            }
        both = all(arm["sets"][s]["mms_ci_entirely_below_syspin"] for s in ("A", "B"))
        arm["verdict"] = ("mms moves ahead of syspin (CI rule met in BOTH sets)" if both
                          else "chain unchanged: the CI rule is not met in both sets")
        arm["rule"] = "challenger moves ahead only if its sentence-CER 95% CI is entirely below the incumbent's in BOTH draw sets"
        result["langs"][lang] = arm
        print(f"{lang}: {arm['verdict']}", flush=True)

    json.dump(mms_rows, open(os.path.join(OUT, "mms_rows.json"), "w", encoding="utf-8"), ensure_ascii=False)
    path = os.path.join(OUT, "bench_mms_vs_syspin.json")
    json.dump(result, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(result["langs"], indent=1, ensure_ascii=False))
    print("report:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
