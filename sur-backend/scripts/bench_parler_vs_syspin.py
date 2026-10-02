"""Step 2: Indic Parler against SYSPIN, under the pre-registered rule.

Protocol identical to docs/tts-benchmark.md, because a comparison is only a
comparison if nothing else moved:
  - the same 20 sentences per language (data/benchmark_sentences.json)
  - the same draw sets, A = 0,1,2 and B = 3,4,5
  - the same Vakyansh reader at the same revision, the same CER, the same
    bootstrap CI over per-line means
  - the SYSPIN arm is NOT re-rendered: its rows come from the committed run
    (.tts_runtime/bench/read.json), so only the challenger costs time

Pre-registered rule, unchanged and not negotiable here: a model moves ahead
of another only if its sentence-CER 95% CI lies entirely below the other's
in BOTH draw sets.

Checkpointed per render, which is not optional at this speed. Parler warmed
up at RTF 44-74 on this CPU (docs/tts-warmup-run.json), so 20 lines x 3
seeds x 2 sets is about 6.7 hours per language. Every finished render is
appended to the checkpoint immediately; a killed run resumes at the next
one instead of starting the language again.

    python scripts/bench_parler_vs_syspin.py --out docs --langs mr,bn
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)
os.environ.setdefault("STORAGE_BACKEND", "local")

DATA = BACKEND / "app" / "tts_runtime" / "data"
BENCH_ROWS = BACKEND / ".tts_runtime" / "bench" / "read.json"
CKPT = "bench_parler_checkpoint.json"


def sentences(lang: str) -> list[str]:
    return json.loads((DATA / "benchmark_sentences.json").read_text(encoding="utf-8"))["langs"][lang]


def save(path: pathlib.Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="mr,bn")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / CKPT
    state = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {"renders": {}}

    from app.tts_runtime import config as cfgmod
    from app.tts_runtime.adapters.vakyansh import VakyanshReader
    from app.tts_runtime.benchmark import SET_A, SET_B, boot_mean_ci
    from app.tts_runtime.checks import cer
    from app.tts_runtime.pool import WorkerError
    from app.tts_runtime.runner import Runner

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    cfg = cfgmod.load(None)
    home = BACKEND / ".tts_runtime"
    runner = Runner(cfg, home)

    reader = VakyanshReader(None, {})
    reader.load("cpu")

    # Harness check: the reader must reproduce a score the committed run
    # recorded, or this is not the benchmark's reader.
    import soundfile as sf
    rows = json.loads(BENCH_ROWS.read_text(encoding="utf-8"))
    ref = next(r for lang in langs for r in rows
               if r.get("lang") == lang and r.get("kind") == "sentence"
               and r.get("cer") is not None and os.path.exists(r.get("path", "")))
    x, sr = sf.read(ref["path"], dtype="float32")
    got = round(cer(ref["text"], reader.read(x, sr, ref["lang"])), 4)
    if abs(got - ref["cer"]) > 1e-4:
        raise SystemExit(f"HARNESS BROKEN: re-scored {ref['path']} as {got}, committed run said {ref['cer']}")
    print(f"harness verified: reader reproduces committed {ref['lang']} score {got}", flush=True)

    wav_dir = home / "bench" / "parler"
    wav_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        texts = sentences(lang)
        for draw in SET_A + SET_B:
            for i, text in enumerate(texts):
                key = f"{lang}:{draw}:{i}"
                if key in state["renders"]:
                    continue
                wav = wav_dir / f"{lang}-{draw}-{i}.wav"
                t0 = time.time()
                try:
                    runner.pool.synth("indic_parler", text, lang, None, None, draw, str(wav))
                    y, ysr = sf.read(str(wav), dtype="float32")
                    hyp = reader.read(y, ysr, lang)
                    row = {"lang": lang, "draw": draw, "set": "A" if draw in SET_A else "B",
                           "text": text, "cer": round(cer(text, hyp), 4), "hyp": hyp,
                           "audio_s": round(len(y) / ysr, 2), "secs": round(time.time() - t0, 1)}
                except WorkerError as e:
                    row = {"lang": lang, "draw": draw, "set": "A" if draw in SET_A else "B",
                           "text": text, "error": f"{e.kind}: {str(e.detail)[:150]}"}
                state["renders"][key] = row
                save(ckpt_path, state)        # after EVERY render
                print(f"  {key}: {row.get('cer', row.get('error'))} ({row.get('secs', '?')}s)", flush=True)
        runner.pool.drop("indic_parler")

    runner.pool.close_all()

    # ── scoring ────────────────────────────────────────────────────────────
    def per_line_means(src, lang, s):
        by = {}
        for r in src:
            if r.get("lang") == lang and r.get("set") == s and r.get("cer") is not None:
                by.setdefault(r["text"], []).append(r["cer"])
        return [sum(v) / len(v) for v in by.values()]

    parler_rows = list(state["renders"].values())
    syspin_rows = [r for r in rows if r.get("kind") == "sentence"]

    result = {"step": "2 benchmark", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "rule": ("challenger moves ahead only if its sentence-CER 95% CI is entirely below the "
                       "incumbent's in BOTH draw sets"),
              "langs": {}}
    for lang in langs:
        arm = {"sets": {}}
        for s in ("A", "B"):
            p = per_line_means(parler_rows, lang, s)
            sy = per_line_means(syspin_rows, lang, s)
            if not p or not sy:
                arm["sets"][s] = {"error": "no rows"}
                continue
            pm, plo, phi = boot_mean_ci(p)
            sm, slo, shi = boot_mean_ci(sy)
            arm["sets"][s] = {
                "indic_parler": {"n_lines": len(p), "mean": round(pm, 4), "ci95": [round(plo, 4), round(phi, 4)]},
                "syspin": {"n_lines": len(sy), "mean": round(sm, 4), "ci95": [round(slo, 4), round(shi, 4)]},
                "parler_ci_entirely_below_syspin": bool(phi < slo),
                "syspin_ci_entirely_below_parler": bool(shi < plo),
            }
        ok = all(arm["sets"].get(s, {}).get("parler_ci_entirely_below_syspin") for s in ("A", "B"))
        arm["verdict"] = ("indic_parler moves ahead of syspin (rule met in BOTH sets)" if ok
                          else "chain unchanged: the rule is not met in both sets")
        errs = [r for r in parler_rows if r.get("lang") == lang and r.get("error")]
        arm["render_errors"] = len(errs)
        arm["median_render_s"] = (sorted(r["secs"] for r in parler_rows if r.get("lang") == lang and r.get("secs"))
                                  or [None])[len([r for r in parler_rows if r.get("lang") == lang and r.get("secs")]) // 2] \
            if any(r.get("secs") for r in parler_rows if r.get("lang") == lang) else None
        result["langs"][lang] = arm
        print(f"{lang}: {arm['verdict']}", flush=True)

    path = out / "tts-benchmark-parler.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"report: {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
