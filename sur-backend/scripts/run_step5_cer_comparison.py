"""Step 5: faster-whisper large-v3 vs Vakyansh vs IndicConformer CER on the 144 hi/mr/bn WAVs.

Evaluates and compares:
1. faster-whisper large-v3 (int8 on CPU)
2. Vakyansh wav2vec2 CTC
3. IndicConformer 600M CTC
Outputs detailed and summary comparison to docs/asr_reader_comparison_144wavs.json.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import soundfile as sf

BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.tts_runtime.adapters.vakyansh import VakyanshReader
from app.tts_runtime.checks import cer
from faster_whisper import WhisperModel


def main():
    docs = BACKEND / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    report_path = docs / "asr_reader_comparison_144wavs.json"
    ckpt_path = docs / "asr_comparison_checkpoint.json"

    # Load IndicConformer results from Step 3 calibration
    conformer_calib_path = docs / "indic_conformer_calibration.json"
    conformer_details = {}
    if conformer_calib_path.exists():
        cdata = json.loads(conformer_calib_path.read_text(encoding="utf-8"))
        for lang, rows in cdata.get("details_by_lang", {}).items():
            for r in rows:
                key = (r["lang"], r["text"], r["draw"])
                conformer_details[key] = r

    bench_read_path = BACKEND / ".tts_runtime" / "bench" / "read.json"
    all_rows = json.loads(bench_read_path.read_text(encoding="utf-8"))
    short_rows = [r for r in all_rows if r.get("kind") == "short" and r.get("lang") in ("hi", "mr", "bn") and r.get("path") and os.path.exists(r["path"])]

    print(f"Total valid short WAVs to benchmark: {len(short_rows)}", flush=True)

    # Load checkpoint if resuming
    state = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {"rows": {}}
    completed_keys = set(state["rows"].keys())

    # Step 5A: Load Vakyansh
    print("Loading Vakyansh reader...", flush=True)
    vakyansh = VakyanshReader(None, {})
    vakyansh.load("cpu")

    # Step 5B: Load faster-whisper large-v3
    print("Loading faster-whisper large-v3...", flush=True)
    whisper = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=4)

    t_start = time.time()
    for i, r in enumerate(short_rows):
        key = f"{r['lang']}:{r['draw']}:{r['text']}"
        if key in completed_keys:
            continue

        wav_path = r["path"]
        ref_text = r["text"]
        lang = r["lang"]
        draw = r.get("draw")

        # 1. IndicConformer (from step 3)
        ckey = (lang, ref_text, draw)
        conformer_entry = conformer_details.get(ckey, {})
        conformer_hyp = conformer_entry.get("hyp", "")
        conformer_cer = conformer_entry.get("cer")

        # 2. Vakyansh
        try:
            x, sr = sf.read(wav_path, dtype="float32")
            vak_hyp = vakyansh.read(x, sr, lang)
            vak_cer = round(cer(ref_text, vak_hyp), 4)
        except Exception as e:
            vak_hyp = f"ERROR: {e}"
            vak_cer = 1.0

        # 3. faster-whisper large-v3
        try:
            segments, _ = whisper.transcribe(
                wav_path,
                language=lang,
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,
                without_timestamps=True,
            )
            fw_hyp = "".join(s.text for s in segments).strip()
            fw_cer = round(cer(ref_text, fw_hyp), 4)
        except Exception as e:
            fw_hyp = f"ERROR: {e}"
            fw_cer = 1.0

        row_result = {
            "lang": lang,
            "draw": draw,
            "text": ref_text,
            "path": wav_path,
            "conformer": {"hyp": conformer_hyp, "cer": conformer_cer},
            "vakyansh": {"hyp": vak_hyp, "cer": vak_cer},
            "faster_whisper": {"hyp": fw_hyp, "cer": fw_cer},
        }
        state["rows"][key] = row_result
        tmp_ckpt = ckpt_path.with_suffix(".tmp")
        tmp_ckpt.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_ckpt, ckpt_path)

        if (i + 1) % 5 == 0 or (i + 1) == len(short_rows):
            print(f"  Processed {i+1}/{len(short_rows)}: {lang} cer(whisper={fw_cer}, conformer={conformer_cer}, vakyansh={vak_cer})", flush=True)

    del vakyansh
    del whisper
    print(f"Completed in {round(time.time() - t_start, 1)}s", flush=True)

    # Aggregate summaries by language
    all_res = list(state["rows"].values())
    summary = {}
    for lang in ("hi", "mr", "bn"):
        lang_items = [x for x in all_res if x["lang"] == lang]
        if not lang_items:
            continue

        c_cers = [x["conformer"]["cer"] for x in lang_items if x["conformer"]["cer"] is not None]
        v_cers = [x["vakyansh"]["cer"] for x in lang_items if x["vakyansh"]["cer"] is not None]
        w_cers = [x["faster_whisper"]["cer"] for x in lang_items if x["faster_whisper"]["cer"] is not None]

        summary[lang] = {
            "n_clips": len(lang_items),
            "indic_conformer": {
                "mean_cer": round(sum(c_cers) / len(c_cers), 4) if c_cers else None,
                "pass_rate_035": round(sum(1 for c in c_cers if c <= 0.35) / len(c_cers), 3) if c_cers else None,
            },
            "vakyansh": {
                "mean_cer": round(sum(v_cers) / len(v_cers), 4) if v_cers else None,
                "pass_rate_035": round(sum(1 for c in v_cers if c <= 0.35) / len(v_cers), 3) if v_cers else None,
            },
            "faster_whisper_large_v3": {
                "mean_cer": round(sum(w_cers) / len(w_cers), 4) if w_cers else None,
                "pass_rate_035": round(sum(1 for c in w_cers if c <= 0.35) / len(w_cers), 3) if w_cers else None,
            },
        }

    report = {
        "step": "5 faster-whisper-vs-vakyansh-vs-indicconformer",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "CER comparison across faster-whisper large-v3, Vakyansh, and IndicConformer on 144 short-line SYSPIN WAVs",
        "summary": summary,
        "details": all_res,
    }

    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Report saved to {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
