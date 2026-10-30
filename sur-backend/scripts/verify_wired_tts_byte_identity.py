"""Run a real project's synthesize stage through both TTS paths and diff.

This is the pipeline's own `synthesize_segment` -- the real Celery task body,
real DB rows, real storage -- run over every translated segment of a real
project, once with TTS_USE_RUNTIME=false (in-process SYSPIN) and once with
TTS_USE_RUNTIME=true (the runtime's isolated subprocess). Both runs' audio is
hashed and compared line by line.

    python scripts/verify_wired_tts_byte_identity.py <project_id> [out_dir]

Needs a project whose segments are already translated (a finished run), the
dev database, and real SYSPIN weights. The evidence it produced for the
wiring change is docs/tts-pipeline-wiring-run.json.

Writes run_report.json: per-line sha256 for each path, equal/not, plus the
provider class actually used, so the report cannot claim a path it did not take.
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


def render_all(project_id: str, use_runtime: bool) -> dict:
    """Synthesize every translated segment of `project_id`, hashing each clip.

    A fresh process would be cleaner still, but the settings cache and the
    provider cache are both cleared here so the flag genuinely decides which
    provider is built.
    """
    os.environ["TTS_USE_RUNTIME"] = "true" if use_runtime else "false"
    os.environ["TTS_PROVIDER"] = "real"
    os.environ["TTS_ENGINE"] = "syspin"

    from app.config import get_settings
    from app.providers import registry

    get_settings.cache_clear()
    registry.get_tts_provider.cache_clear()
    settings = get_settings()
    assert settings.tts_use_runtime is use_runtime, "the flag did not take effect"

    from app.db import session_scope
    from app.models import Project, Segment
    from app.pipeline.tasks import synthesize_segment
    from app.storage import get_storage

    storage = get_storage()
    provider_name = type(registry.get_tts_provider()).__name__
    out = {"provider": provider_name, "use_runtime": use_runtime, "lines": {}}
    t0 = time.time()
    with session_scope() as db:
        project = db.get(Project, project_id)
        assert project is not None, f"no project {project_id}"
        segs = (db.query(Segment)
                  .filter(Segment.project_id == project_id, Segment.translated_text.isnot(None))
                  .order_by(Segment.index).all())
        assert segs, "project has no translated segments"
        for seg in segs:
            synthesize_segment(db, storage, seg, project)
            key = seg.tts_audio_url
            if not key:
                out["lines"][seg.index] = {"text": seg.translated_text, "sha256": None, "skipped": True}
                continue
            import tempfile

            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                storage.download_file(key, tmp)
                data = open(tmp, "rb").read()
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            out["lines"][seg.index] = {
                "text": seg.translated_text,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "duration_ms": seg.tts_duration_ms,
            }
        db.rollback()      # leave the project's rows exactly as they were
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> int:
    project_id = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    off = render_all(project_id, use_runtime=False)
    on = render_all(project_id, use_runtime=True)

    lines = []
    for idx in sorted(off["lines"], key=int):
        a, b = off["lines"][idx], on["lines"].get(idx, {})
        lines.append({"index": idx, "text": a["text"],
                      "in_process_sha256": a["sha256"], "runtime_sha256": b.get("sha256"),
                      "equal": a["sha256"] == b.get("sha256"),
                      "duration_ms": a.get("duration_ms")})
    report = {
        "project": project_id,
        "in_process": {"provider": off["provider"], "seconds": off["seconds"]},
        "runtime": {"provider": on["provider"], "seconds": on["seconds"]},
        "lines": lines,
        "lines_total": len(lines),
        "lines_byte_identical": sum(1 for x in lines if x["equal"]),
    }
    path = os.path.join(out_dir, "run_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "lines"}, indent=1, ensure_ascii=False))
    for x in lines:
        print(f"  [{x['index']}] {'OK ' if x['equal'] else 'DIFF'} {x['text'][:38]!r} "
              f"{(x['in_process_sha256'] or '-')[:12]} vs {(x['runtime_sha256'] or '-')[:12]}")
    print("report:", path)
    return 0 if report["lines_byte_identical"] == report["lines_total"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
