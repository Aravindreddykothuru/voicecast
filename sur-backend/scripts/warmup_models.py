"""Step 1: warm up one model at a time and record what it costs.

One model is loaded, every language it supports speaks one sentence, then the
worker is killed before the next model loads. That ordering is the point: on
a 15 GB box with ~2 GB free, two Indic models resident at once is an OOM, and
an OOM mid-benchmark is a measurement you cannot trust.

Resumable on purpose. Every (model, language) pair is appended to a
checkpoint as soon as it finishes, so a killed run continues from the next
pair instead of re-loading 3.7 GB of weights to redo work already done.

Peak RAM is the worker's own PeakWorkingSetSize (worker.py::_rss_mb), i.e.
the high-water mark of the process that actually held the weights, not a
sample of the parent taken at some arbitrary moment.

    python scripts/warmup_models.py --out docs --models indic_parler
    python scripts/warmup_models.py --out docs            # every chain model
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

CHECKPOINT = "warmup_checkpoint.json"


def load_checkpoint(path: pathlib.Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"rows": [], "done": []}


def save(path: pathlib.Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="", help="comma-separated; default every model in the chains")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / CHECKPOINT
    state = load_checkpoint(ckpt_path)
    done = {tuple(d) for d in state["done"]}

    from app.tts_runtime import config as cfgmod
    from app.tts_runtime.adapters import adapter_class
    from app.tts_runtime.checks import signal_checks
    from app.tts_runtime.pool import WorkerError, WorkerOOM
    from app.tts_runtime.provision import WARMUP_TEXT
    from app.tts_runtime.runner import Runner

    cfg = cfgmod.load(args.config)
    home = BACKEND / ".tts_runtime"
    sentences = json.loads(pathlib.Path(WARMUP_TEXT).read_text(encoding="utf-8"))

    wanted = [m.strip() for m in args.models.split(",") if m.strip()] or None
    pairs: list[tuple[str, str]] = []
    for lang in sorted(set(cfg.chains) | set(sentences)):
        for model in cfg.chain_for(lang):
            if wanted and model not in wanted:
                continue
            if (model, lang) not in pairs:
                pairs.append((model, lang))

    by_model: dict[str, list[str]] = {}
    for model, lang in pairs:
        by_model.setdefault(model, []).append(lang)

    runner = Runner(cfg, home)   # builds its own Store and pool
    print(f"models: {list(by_model)}", flush=True)

    for model, langs in by_model.items():
        cls = adapter_class(model)
        print(f"\n=== {model}: {langs} ===", flush=True)
        for lang in langs:
            if (model, lang) in done:
                print(f"  {lang}: already done, skipping", flush=True)
                continue
            row: dict = {"model": model, "lang": lang}

            if not cls.supports(lang):
                row["result"] = "unsupported"
            else:
                text = sentences.get(lang)
                ok, why = cls.availability(runner.options_for(model))
                if not text:
                    row.update(result="no test sentence")
                elif not ok:
                    row.update(result="unavailable", why=why)
                else:
                    row.update(_one(runner, model, lang, text, home, signal_checks, WorkerError, WorkerOOM))

            state["rows"].append(row)
            state["done"].append([model, lang])
            done.add((model, lang))
            save(ckpt_path, state)   # after EVERY pair, so a kill resumes here
            print(f"  {lang}: {json.dumps({k: v for k, v in row.items() if k not in ('model', 'lang')}, ensure_ascii=False)}", flush=True)

        # Unload before the next model: two Indic models resident is an OOM here.
        runner.pool.drop(model)
        print(f"  unloaded {model}", flush=True)

    runner.pool.close_all()

    report = {
        "step": "1 warm-up",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Windows 11, Ryzen 5 7520U, CPU only, 15.24 GB RAM",
        "method": ("one model loaded at a time and killed before the next; peak RAM is the worker's own "
                   "PeakWorkingSetSize, not a sample of the parent"),
        "rows": state["rows"],
    }
    path = out / "tts-warmup-run.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport: {path}", flush=True)
    return 0


def _one(runner, model, lang, text, home, signal_checks, WorkerError, WorkerOOM) -> dict:
    """Load (if needed), speak one sentence, report cost. Retries once on OOM."""
    import soundfile as sf

    out_wav = home / "warmup" / f"{model.replace(':', '_')}-{lang}.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    for attempt in (1, 2):
        t0 = time.time()
        try:
            r = runner.pool.synth(model, text, lang, None, None, 0, str(out_wav))
        except WorkerOOM as e:
            st = runner.pool.stats.get(model, {})
            if attempt == 1:
                # Record what it reached before dying, drop it, try once more
                # with nothing else of ours resident.
                print(f"  {lang}: OOM (peak {st.get('rss_mb')} MB) -- dropping and retrying once", flush=True)
                runner.pool.drop(model)
                continue
            return {"result": "oom", "why": str(e.detail)[:200], "peak_rss_mb_before_oom": st.get("rss_mb"),
                    "attempts": attempt}
        except WorkerError as e:
            return {"result": "fail", "why": f"{e.kind}: {str(e.detail)[:200]}",
                    "secs": round(time.time() - t0, 1), "attempts": attempt}

        x, sr = sf.read(str(out_wav), dtype="float32")
        chk = signal_checks(x, sr, text, lang, runner.calibration)
        st = runner.pool.stats.get(model, {})
        return {
            "result": "pass" if chk.passed else "fail",
            "why": "; ".join(f["detail"] for f in chk.failures),
            "load_s": st.get("load_s"),
            "first_synth_s": round(r.get("secs", 0), 2),
            "peak_rss_mb": r.get("rss_mb"),
            "vram_mb": r.get("vram_mb"),
            "device": st.get("device"),
            "audio_s": round(len(x) / sr, 2),
            "attempts": attempt,
        }
    return {"result": "oom", "attempts": 2}


if __name__ == "__main__":
    sys.exit(main())
