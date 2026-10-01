"""Steady-state per-line cost: in-process SYSPIN vs the runtime subprocess.

The earlier e2e number (68.4 s vs 163.1 s over 9 lines) was NOT comparable:
the in-process provider had pre-loaded every voice during its startup
self-check, outside the measured window, while the runtime worker loaded
lazily inside it. This measures the steady state instead -- both providers
warmed first, then 9 lines x 3 runs each, interleaved so CPU drift hits both
equally.

    python measure_runtime_overhead.py [out_dir]

Reports per-line mean and spread for synthesize() (what production pays) and
for _render_raw() alone (the only step the flag changes).
"""
import json
import os
import pathlib
import statistics
import sys
import time

BACKEND = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)
os.environ.setdefault("TTS_PROVIDER", "real")
os.environ.setdefault("TTS_ENGINE", "syspin")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RUNS = 3
OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# The same nine real lines as docs/tts-pipeline-wiring-run.json, so the two
# measurements describe the same work.
LINES = [ln["text"] for ln in json.load(
    open(os.path.join(BACKEND, "docs/tts-pipeline-wiring-run.json"), encoding="utf-8"))["lines"]]
LANG = "te"


def build():
    from app.providers.tts.runtime_syspin_provider import RuntimeSyspinTTSProvider
    from app.providers.tts.syspin_provider import SyspinTTSProvider

    t0 = time.perf_counter()
    in_proc = SyspinTTSProvider()
    t1 = time.perf_counter()
    runtime = RuntimeSyspinTTSProvider()
    t2 = time.perf_counter()
    return in_proc, runtime, {"in_process_construct_s": round(t1 - t0, 1),
                              "runtime_construct_s": round(t2 - t1, 1)}


def one_line(provider, text):
    """Seconds for synthesize(), and for the raw render inside it."""
    from app.capabilities import require_language
    from app.providers.base import SynthesisRequest

    raw = {}
    original = type(provider)._render_raw

    def timed(self, voice, t, lang, gender):
        s = time.perf_counter()
        out = original(self, voice, t, lang, gender)
        raw["s"] = time.perf_counter() - s
        return out

    type(provider)._render_raw = timed
    try:
        req = SynthesisRequest(text=text, target_lang=LANG, target_duration_ms=8000)
        s = time.perf_counter()
        res = provider.synthesize(req)
        total = time.perf_counter() - s
    finally:
        type(provider)._render_raw = original
    try:
        os.remove(res.local_audio_path)
    except OSError:
        pass
    require_language(LANG)
    return total, raw["s"]


def main() -> int:
    in_proc, runtime, construct = build()
    print(json.dumps(construct), flush=True)

    # Warm both: the first line of each pays voice loading (in the parent for
    # one, in the worker for the other). Neither warm-up is measured.
    for p in (in_proc, runtime):
        one_line(p, LINES[0])
    print("warmed", flush=True)

    samples = {"in_process": {"total": [], "raw": []}, "runtime": {"total": [], "raw": []}}
    for run in range(RUNS):
        for name, p in (("in_process", in_proc), ("runtime", runtime)):   # interleaved
            for text in LINES:
                total, raw = one_line(p, text)
                samples[name]["total"].append(total)
                samples[name]["raw"].append(raw)
            print(f"run {run + 1} {name} done", flush=True)

    def stats(v):
        return {"n": len(v), "mean_s": round(statistics.mean(v), 3),
                "sd_s": round(statistics.stdev(v), 3) if len(v) > 1 else 0.0,
                "min_s": round(min(v), 3), "max_s": round(max(v), 3),
                "total_s": round(sum(v), 1)}

    report = {"lines": len(LINES), "runs": RUNS, "lang": LANG, "construct": construct,
              "synthesize": {k: stats(samples[k]["total"]) for k in samples},
              "render_raw": {k: stats(samples[k]["raw"]) for k in samples}}
    a, b = report["synthesize"]["in_process"], report["synthesize"]["runtime"]
    report["overhead_per_line_s"] = round(b["mean_s"] - a["mean_s"], 3)
    report["overhead_pct"] = round(100 * (b["mean_s"] - a["mean_s"]) / a["mean_s"], 1)
    ra, rb = report["render_raw"]["in_process"], report["render_raw"]["runtime"]
    report["render_raw_overhead_per_line_s"] = round(rb["mean_s"] - ra["mean_s"], 3)

    path = os.path.join(OUT, "runtime_overhead.json")
    json.dump(report, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(report, indent=1))
    print("report:", path)
    runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
