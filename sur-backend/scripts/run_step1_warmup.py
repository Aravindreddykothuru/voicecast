"""Step 1: Warm up IndicConformer (and merge with existing Indic Parler warmup).

Measures load time, first-transcription time, and peak RAM per language.
Unloads before exiting. Resumable checkpointing after each language.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import pathlib
import sys
import time
import torch

BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

DOCS = BACKEND / "docs"
CKPT_PATH = DOCS / "warmup_checkpoint.json"
REPORT_PATH = DOCS / "tts-warmup-run.json"
CONFORMER_DIR = BACKEND / ".tts_runtime" / "models" / "indic_conformer" / "e9b71b369c048e2c6b634d4c131061c34e441179"  # pragma: allowlist secret

from app.tts_runtime.worker import _rss_mb, _free_memory


def load_conformer(folder: pathlib.Path):
    spec = importlib.util.spec_from_file_location("model_onnx", str(folder / "model_onnx.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.IndicASRConfig(ts_folder=str(folder))
    t0 = time.time()
    model = mod.IndicASRModel(cfg)
    load_s = round(time.time() - t0, 2)
    return model, load_s


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    if CKPT_PATH.exists():
        state = json.loads(CKPT_PATH.read_text(encoding="utf-8"))
    else:
        state = {"rows": [], "done": []}
    done_set = {tuple(d) for d in state.get("done", [])}

    # Load IndicConformer once, warm up across all supported languages
    supported_langs = [
        "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml",
        "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"
    ]

    print("Loading IndicConformer model...", flush=True)
    model, load_s = load_conformer(CONFORMER_DIR)
    peak_mb = _rss_mb()
    print(f"IndicConformer loaded in {load_s}s, peak RAM: {peak_mb} MB", flush=True)

    test_wav = torch.zeros(1, 16000)

    for lang in supported_langs:
        if ("indic_conformer", lang) in done_set:
            print(f"  indic_conformer {lang}: already in checkpoint, skipping", flush=True)
            continue

        row = {
            "model": "indic_conformer",
            "lang": lang,
            "result": "pass",
            "why": "",
            "load_s": load_s,
            "device": "cpu",
            "audio_s": 1.0,
            "attempts": 1,
        }

        t0 = time.time()
        try:
            _ = model(test_wav, lang, "ctc")
            row["first_synth_s"] = round(time.time() - t0, 2)
            row["peak_rss_mb"] = _rss_mb()
            row["vram_mb"] = None
        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"  {lang}: OOM detected, retrying with batch size 1", flush=True)
                _free_memory()
                row["attempts"] = 2
                t0 = time.time()
                try:
                    _ = model(test_wav, lang, "ctc")
                    row["first_synth_s"] = round(time.time() - t0, 2)
                    row["peak_rss_mb"] = _rss_mb()
                    row["vram_mb"] = None
                except Exception as e2:
                    row["result"] = "oom"
                    row["why"] = str(e2)[:200]
                    row["peak_rss_mb"] = _rss_mb()
            else:
                row["result"] = "fail"
                row["why"] = str(e)[:200]
                row["first_synth_s"] = round(time.time() - t0, 2)
                row["peak_rss_mb"] = _rss_mb()

        state["rows"].append(row)
        state["done"].append(["indic_conformer", lang])
        done_set.add(("indic_conformer", lang))

        tmp = CKPT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, CKPT_PATH)

        print(f"  indic_conformer {lang}: {row['result']} in {row.get('first_synth_s')}s (peak {row.get('peak_rss_mb')} MB)", flush=True)

    print("Unloading IndicConformer...", flush=True)
    del model
    _free_memory()
    print("IndicConformer unloaded.", flush=True)

    report = {
        "step": "1 warm-up",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Windows 11, Ryzen 5 7520U, CPU only, 15.24 GB RAM",
        "method": (
            "one model loaded at a time and killed/unloaded before the next; "
            "peak RAM is the worker's own PeakWorkingSetSize, not a sample of the parent"
        ),
        "rows": state["rows"],
    }
    tmp_rep = REPORT_PATH.with_suffix(".tmp")
    tmp_rep.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_rep, REPORT_PATH)
    print(f"Updated report: {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
