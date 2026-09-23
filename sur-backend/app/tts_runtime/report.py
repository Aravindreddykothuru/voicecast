"""The run report: everything the runtime did, counted from the store.

No silent fallbacks: every model switch, retry, pause, resume, breaker event,
OOM, known-bad addition, flagged line and unavailable model is an event, and
every event kind is counted here. Written next to the job's outputs as
report.json and report.md.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.tts_runtime.fsutil import atomic_write_bytes

COUNTED = ("switch", "retry", "pause", "resume", "breaker_open", "breaker_close", "breaker_half_open", "oom",
           "oom_cpu_fallback", "unload", "known_bad_added", "known_bad_last_resort", "model_unavailable",
           "model_corrupt", "restart_recovered", "disk_full", "reader_error", "unverified", "scene_rerender",
           "scene_mixed", "task_done", "task_failed", "hash_mismatch", "health_check", "stop_requested",
           "flagged")


def build(store, job_id: str, pool_status: dict | None = None) -> dict:
    job = store.job(job_id)
    lines = store.lines(job_id)
    t0 = job["created_at"]
    evs = [e for e in store.q("SELECT * FROM events WHERE job_id=? OR (job_id IS NULL AND at>=?) ORDER BY id",
                              (job_id, t0))]
    kinds = Counter(e["kind"] for e in evs)
    attempts = store.attempts(job_id)
    outcomes = Counter(a["outcome"] for a in attempts)
    done_by_model = Counter(ln["model"] for ln in lines if ln["state"] == "DONE")
    flagged = [{"idx": ln["idx"], "lang": ln["lang"], "text": ln["text"], "kept_model": ln["model"],
                "reason": ln["flag_reason"], "has_audio": ln["output_path"] is not None}
               for ln in lines if ln["state"] == "FLAGGED"]
    breaker_events = [{"at": e["at"], "kind": e["kind"], "model": e["model"], "lang": e["lang"], "reason": e["detail"]}
                      for e in evs if e["kind"].startswith("breaker_")]
    unavailable = sorted({(e["model"], e["detail"]) for e in evs if e["kind"] == "model_unavailable"})
    return {
        "job": job_id, "status": job["status"], "lines": len(lines), "states": store.counts(job_id),
        "lines_per_model": dict(done_by_model), "flagged": flagged,
        "switches": kinds["switch"], "retries": kinds["retry"],
        "pauses": kinds["pause"], "resumes": kinds["resume"],
        "circuit_breaker_events": breaker_events,
        "events": {k: kinds[k] for k in COUNTED if kinds[k]},
        "attempt_outcomes": dict(outcomes),
        "cache_hits": outcomes["cache_hit"],
        "unavailable_models": [{"model": m, "why": w} for m, w in unavailable],
        "known_bad_added": [json.loads(e["detail"]) for e in evs if e["kind"] == "known_bad_added"],
        "test_models_enabled": any(json.loads(e["detail"] or "{}").get("test_models") for e in evs
                                   if e["kind"] == "job_started"),
        "models": (pool_status or {}).get("stats", {}),
    }


def markdown(r: dict) -> str:
    out = [f"# TTS run report: job {r['job']}", "",
           f"Status **{r['status']}** -- {r['lines']} lines: " +
           ", ".join(f"{k} {v}" for k, v in r["states"].items() if v), ""]
    if r["test_models_enabled"]:
        out += ["> **Test models were enabled for this run.** Not a production run.", ""]
    out += ["| lines per model | |", "|---|---|"] + [f"| {m} | {n} |" for m, n in sorted(r["lines_per_model"].items())]
    out += ["", f"Switches {r['switches']} · retries {r['retries']} · pauses {r['pauses']} · resumes {r['resumes']} · "
            f"cache hits {r['cache_hits']}", ""]
    if r["events"]:
        out += ["| event | count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in r["events"].items()] + [""]
    if r["unavailable_models"]:
        out += ["## Models unavailable", ""] + [f"- `{u['model']}`: {u['why']}" for u in r["unavailable_models"]] + [""]
    if r["circuit_breaker_events"]:
        out += ["## Circuit breaker", ""] + [f"- {e['kind']} `{e['model']}` / {e['lang']}: {e['reason']}"
                                             for e in r["circuit_breaker_events"]] + [""]
    out += [f"## Flagged for human review ({len(r['flagged'])})", ""]
    out += [f"- line {f['idx']} [{f['lang']}] {f['text']!r}: {f['reason']}"
            + ("" if f["has_audio"] else " -- **no audio**") for f in r["flagged"]] or ["none"]
    if r["known_bad_added"]:
        out += ["", "## Known-bad words learned", ""] + [f"- {k['word']}: {k['evidence']}" for k in r["known_bad_added"]]
    return "\n".join(out) + "\n"


def write_report(store, job_id: str, out_dir: Path, pool_status: dict | None = None) -> dict:
    r = build(store, job_id, pool_status)
    out_dir = Path(out_dir)
    atomic_write_bytes(out_dir / "report.json", json.dumps(r, indent=1, ensure_ascii=False, default=str).encode())
    atomic_write_bytes(out_dir / "report.md", markdown(r).encode("utf-8"))
    return r
