"""Step 3: IndicConformer reader calibration for hi/mr/bn and re-check of क्यों and रुकिए.

Evaluates IndicConformer (CTC mode) against the 144 short-line SYSPIN audio renders.
Specifically analyzes performance on known-bad candidates (क्यों, रुकिए) and measures
trigger viability (pass rate at CER <= 0.35).
Outputs evidence to docs/indic_conformer_calibration.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import time
import soundfile as sf
import torch
import torchaudio

BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.tts_runtime.checks import cer
CONFORMER_DIR = BACKEND / ".tts_runtime" / "models" / "indic_conformer" / "e9b71b369c048e2c6b634d4c131061c34e441179"  # pragma: allowlist secret


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
    docs = BACKEND / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    report_path = docs / "indic_conformer_calibration.json"

    bench_read_path = BACKEND / ".tts_runtime" / "bench" / "read.json"
    rows = json.loads(bench_read_path.read_text(encoding="utf-8"))
    short_rows = [r for r in rows if r.get("kind") == "short" and r.get("lang") in ("hi", "mr", "bn") and r.get("path") and os.path.exists(r["path"])]

    print(f"Loaded {len(short_rows)} valid short-line WAV rows for hi/mr/bn", flush=True)

    print("Loading IndicConformer...", flush=True)
    model, load_s = load_conformer(CONFORMER_DIR)
    print(f"IndicConformer loaded in {load_s}s", flush=True)

    results_by_lang = {"hi": [], "mr": [], "bn": []}
    kyon_rukiye_results = []

    t_start = time.time()
    for i, r in enumerate(short_rows):
        wav_path = r["path"]
        ref_text = r["text"]
        lang = r["lang"]
        draw = r.get("draw")

        # Load audio and resample to 16000 if needed
        data, sr = sf.read(wav_path, dtype="float32")
        x = torch.from_numpy(data)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            x = x.mean(dim=-1, keepdim=True).t()

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            x = resampler(x)

        # Transcribe with IndicConformer
        try:
            hyp_text = model(x, lang, "ctc")
        except Exception as e:
            hyp_text = f"ERROR: {e}"

        c = round(cer(ref_text, hyp_text), 4)

        entry = {
            "lang": lang,
            "draw": draw,
            "text": ref_text,
            "hyp": hyp_text,
            "cer": c,
            "vakyansh_cer": r.get("cer"),
            "path": wav_path,
        }
        results_by_lang[lang].append(entry)

        # Track क्यों and रुकिए specifically
        if ref_text in ("क्यों?", "रुकिए।") or "क्यों" in ref_text or "रुकिए" in ref_text:
            kyon_rukiye_results.append(entry)

        if (i + 1) % 20 == 0 or (i + 1) == len(short_rows):
            print(f"  Processed {i+1}/{len(short_rows)} WAVs...", flush=True)

    del model
    print(f"Transcription finished in {round(time.time() - t_start, 1)}s", flush=True)

    # Compute language-level statistics
    summary = {}
    for lang, items in results_by_lang.items():
        cers = [x["cer"] for x in items]
        vak_cers = [x["vakyansh_cer"] for x in items if x["vakyansh_cer"] is not None]
        passed_035 = sum(1 for c in cers if c <= 0.35)
        vak_passed_035 = sum(1 for c in vak_cers if c <= 0.35)

        summary[lang] = {
            "n_clips": len(items),
            "conformer_mean_cer": round(sum(cers) / len(cers), 4),
            "conformer_pass_rate_035": round(passed_035 / len(items), 3),
            "vakyansh_mean_cer": round(sum(vak_cers) / len(vak_cers), 4) if vak_cers else None,
            "vakyansh_pass_rate_035": round(vak_passed_035 / len(vak_cers), 3) if vak_cers else None,
            "cer_improvement": round((sum(vak_cers) / len(vak_cers)) - (sum(cers) / len(cers)), 4) if vak_cers else None,
        }

    # Analyze क्यों and रुकिए specifically
    kyon_entries = [x for x in kyon_rukiye_results if "क्यों" in x["text"]]
    rukiye_entries = [x for x in kyon_rukiye_results if "रुकिए" in x["text"]]

    word_analysis = {
        "क्यों": {
            "n": len(kyon_entries),
            "conformer_mean_cer": round(sum(x["cer"] for x in kyon_entries) / max(len(kyon_entries), 1), 4),
            "vakyansh_mean_cer": round(sum(x["vakyansh_cer"] for x in kyon_entries if x["vakyansh_cer"] is not None) / max(len(kyon_entries), 1), 4),
            "samples": [{"draw": x["draw"], "hyp": x["hyp"], "conformer_cer": x["cer"], "vakyansh_cer": x["vakyansh_cer"]} for x in kyon_entries],
        },
        "रुकिए": {
            "n": len(rukiye_entries),
            "conformer_mean_cer": round(sum(x["cer"] for x in rukiye_entries) / max(len(rukiye_entries), 1), 4),
            "vakyansh_mean_cer": round(sum(x["vakyansh_cer"] for x in rukiye_entries if x["vakyansh_cer"] is not None) / max(len(rukiye_entries), 1), 4),
            "samples": [{"draw": x["draw"], "hyp": x["hyp"], "conformer_cer": x["cer"], "vakyansh_cer": x["vakyansh_cer"]} for x in rukiye_entries],
        }
    }

    report = {
        "step": "3 conformer-reader-calibration",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "IndicConformer 600M CTC inference on SYSPIN short lines (draws 0-5), compared to Vakyansh ground truth",
        "summary": summary,
        "word_recheck": word_analysis,
        "details_by_lang": results_by_lang,
    }

    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Calibration report written to {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
