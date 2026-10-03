"""Export current Step 5 CER comparison checkpoint to docs/asr_reader_comparison_144wavs.json."""
import json
import time
from pathlib import Path

ckpt_path = Path("docs/asr_comparison_checkpoint.json")
report_path = Path("docs/asr_reader_comparison_144wavs.json")

if not ckpt_path.exists():
    print("Checkpoint does not exist yet.")
    raise SystemExit(1)

ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
rows = list(ckpt["rows"].values())

summary = {}
for lang in ("hi", "mr", "bn"):
    items = [x for x in rows if x["lang"] == lang]
    if not items:
        continue
    c = [x["conformer"]["cer"] for x in items if x["conformer"]["cer"] is not None]
    v = [x["vakyansh"]["cer"] for x in items if x["vakyansh"]["cer"] is not None]
    w = [x["faster_whisper"]["cer"] for x in items if x["faster_whisper"]["cer"] is not None]
    summary[lang] = {
        "n_clips": len(items),
        "indic_conformer": {
            "mean_cer": round(sum(c) / len(c), 4) if c else None,
            "pass_rate_035": round(sum(1 for x in c if x <= 0.35) / len(c), 3) if c else None,
        },
        "vakyansh": {
            "mean_cer": round(sum(v) / len(v), 4) if v else None,
            "pass_rate_035": round(sum(1 for x in v if x <= 0.35) / len(v), 3) if v else None,
        },
        "faster_whisper_large_v3": {
            "mean_cer": round(sum(w) / len(w), 4) if w else None,
            "pass_rate_035": round(sum(1 for x in w if x <= 0.35) / len(w), 3) if w else None,
        },
    }

report = {
    "step": "5 faster-whisper-vs-vakyansh-vs-indicconformer",
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "method": "CER comparison across faster-whisper large-v3, Vakyansh, and IndicConformer on 144 short-line SYSPIN WAVs",
    "total_processed": len(rows),
    "summary": summary,
    "details": rows,
}

report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Exported {len(rows)} rows to {report_path}")
print(json.dumps(summary, indent=2))
