"""`tts` -- the runtime's command line.

  tts setup [--timeout S]        download + verify every pinned model file (idempotent)
  tts warmup [--langs hi,te]     every model x language speaks one sentence
  tts run SPEC.json              create a job from a spec and run it
  tts resume JOB_ID              continue an interrupted job from exactly where it stopped
  tts supervise SPEC.json|JOB    run under the watchdog: auto-restart and auto-resume until DONE
  tts status [JOB_ID]            line counts, heartbeat, network tasks
  tts report JOB_ID              the run report (markdown)
  tts benchmark [--models m,..] [--langs hi,..]   evidence for chain order + trigger calibration

A spec: {"job_id": optional, "out_dir": optional,
         "lines": [{"text", "lang", "speaker"?, "emotion"?, "scene"?}, ...]}
Exit codes: 0 job DONE, 3 interrupted (resume to continue), 1 error.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from app.tts_runtime import config as cfgmod
from app.tts_runtime.adapters.base import runtime_home

OPTIONS_ENV = "TTS_RUNTIME_MODEL_OPTIONS"


def _runner(a, **kw):
    """Per-model options from $TTS_RUNTIME_MODEL_OPTIONS (JSON) are honoured only
    with test models enabled: they exist to drive fake models in chaos tests."""
    import os

    from app.tts_runtime.runner import Runner

    cfg = cfgmod.load(a.config)
    if os.environ.get(OPTIONS_ENV):
        if not cfg.allow_test_models:
            raise cfgmod.ConfigError(f"{OPTIONS_ENV} is only honoured with {cfgmod.ALLOW_TEST_MODELS_ENV}=1")
        kw.setdefault("options", json.loads(os.environ[OPTIONS_ENV]))
    return Runner(cfg, a.home, **kw)


def cmd_setup(a) -> int:
    from app.tts_runtime.netmon import ConnectivityMonitor
    from app.tts_runtime.provision import setup
    from app.tts_runtime.store import Store

    cfg = cfgmod.load(a.config)
    # start() the monitor, as Runner does. Without it nothing polls during a
    # setup: connectivity is only re-evaluated when a download fails and calls
    # check_now(), so once every task is PAUSED (offline) there is no longer
    # anything left to fail, nothing asks again, and the queue waits forever
    # for news that never arrives. Observed twice on a 3.7 GB download -- the
    # link came back within seconds and setup sat paused for over an hour.
    netmon = ConnectivityMonitor(cfg.offline_mode)
    netmon.start()
    try:
        rep = setup(cfg, a.home, Store(a.home / "runtime.sqlite3"), netmon, timeout_s=a.timeout)
    finally:
        netmon.stop()
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    failed = [t for t, v in rep["tasks"].items() if v["state"] == "FAILED"]
    # A model named in a chain but left unprovisioned is a half-set-up
    # deployment, not a success: every line for it would be skipped at render
    # time. Say which one and why, and exit non-zero -- never "continue
    # silently with the one model that happened to work".
    unprovisioned = {m: v["skipped_gated"] for m, v in rep["models"].items()
                     if isinstance(v, dict) and v.get("skipped_gated")}
    if unprovisioned or failed:
        for model, repos in unprovisioned.items():
            print(f"tts setup: {model} NOT provisioned -- {', '.join(repos)} is gated and HF_TOKEN is "
                  f"{'not set' if not rep['token'] else 'set but was refused'}. Accept the licence on "
                  f"huggingface.co, export HF_TOKEN, and re-run: it resumes from what is already downloaded.",
                  file=sys.stderr)
        for t in failed:
            print(f"tts setup: task {t} FAILED -- {rep['tasks'][t]['error']}", file=sys.stderr)
        return 1
    return 0


def cmd_warmup(a) -> int:
    from app.tts_runtime.provision import warmup

    r = _runner(a)
    try:
        rows = warmup(r, a.langs.split(",") if a.langs else None)
    finally:
        r.pool.close_all()
    (a.home / "warmup.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{'model':14s} {'lang':4s} {'result':12s} {'load s':>7s} {'synth s':>8s} {'rss MB':>7s} {'vram MB':>8s}  why")
    for x in rows:
        print(f"{x['model']:14s} {x['lang']:4s} {x['result']:12s} {str(x.get('load_s') or ''):>7s} "
              f"{str(x.get('synth_s') or ''):>8s} {str(x.get('rss_mb') or ''):>7s} {str(x.get('vram_mb') or ''):>8s}  "
              f"{x.get('why', '')[:90]}")
    return 1 if any(x["result"] == "fail" for x in rows) else 0


def _finish(res: dict) -> int:
    print(json.dumps(res, indent=1, default=str))
    return 0 if res["status"] == "DONE" else 3


def cmd_run(a) -> int:
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    r = _runner(a, health_port=a.health_port)
    jid = r.submit(spec)
    print(f"job {jid}", flush=True)
    return _finish(r.run(jid))


def cmd_resume(a) -> int:
    r = _runner(a, health_port=a.health_port)
    return _finish(r.run(a.job_id))


def cmd_supervise(a) -> int:
    from app.tts_runtime.supervisor import supervise

    target = Path(a.target)
    if target.suffix == ".json" and target.exists():
        r = _runner(a)
        jid = r.submit(json.loads(target.read_text(encoding="utf-8")))
        r.store.close()
    else:
        jid = a.target
    print(f"supervising job {jid}", flush=True)
    return supervise(jid, a.home, config=str(a.config) if a.config else None, stale_s=a.stale_s)


def cmd_status(a) -> int:
    from app.tts_runtime.store import Store

    s = Store(a.home / "runtime.sqlite3")
    hb_path = a.home / "heartbeat.json"
    hb = json.loads(hb_path.read_text(encoding="utf-8")) if hb_path.exists() else None
    jobs = [s.job(a.job_id)] if a.job_id else s.q("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 10")
    out = {"heartbeat": hb, "heartbeat_age_s": round(time.time() - hb["at"], 1) if hb else None,
           "network_tasks": {st: len(s.tasks((st,))) for st in ("QUEUED", "RUNNING", "PAUSED", "DONE", "FAILED")},
           "jobs": [{"id": j["id"], "status": j["status"], "pause_reason": j["pause_reason"], "lines": s.counts(j["id"])}
                    for j in jobs if j]}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


def cmd_report(a) -> int:
    from app.tts_runtime.report import build, markdown
    from app.tts_runtime.store import Store

    print(markdown(build(Store(a.home / "runtime.sqlite3"), a.job_id)))
    return 0


def cmd_benchmark(a) -> int:
    from app.tts_runtime import benchmark

    r = _runner(a)
    models = a.models.split(",") if a.models else r.cfg.models()
    langs = a.langs.split(",") if a.langs else sorted(r.cfg.chains)
    try:
        res = benchmark.run(r, models, langs, write_calibration=not a.no_calibration)
    except benchmark.HarnessError as e:
        print(f"tts benchmark: HARNESS BROKEN, no result is trustworthy: {e}", file=sys.stderr)
        return 2
    finally:
        r.pool.close_all()
    print(benchmark.markdown(res))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tts", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None, help="tts_chains.yaml (default: sur-backend/tts_chains.yaml)")
    ap.add_argument("--home", type=Path, default=None, help="runtime state dir (default: $TTS_RUNTIME_HOME or .tts_runtime)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--timeout", type=float, default=None)
    p.set_defaults(fn=cmd_setup)
    p = sub.add_parser("warmup")
    p.add_argument("--langs", default=None)
    p.set_defaults(fn=cmd_warmup)
    p = sub.add_parser("run")
    p.add_argument("spec")
    p.add_argument("--health-port", type=int, default=None)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("resume")
    p.add_argument("job_id")
    p.add_argument("--health-port", type=int, default=None)
    p.set_defaults(fn=cmd_resume)
    p = sub.add_parser("supervise")
    p.add_argument("target")
    p.add_argument("--stale-s", type=float, default=180.0)
    p.set_defaults(fn=cmd_supervise)
    p = sub.add_parser("status")
    p.add_argument("job_id", nargs="?")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("report")
    p.add_argument("job_id")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("benchmark")
    p.add_argument("--models", default=None)
    p.add_argument("--langs", default=None)
    p.add_argument("--no-calibration", action="store_true", help="do not overwrite app/tts_runtime/calibration.json")
    p.set_defaults(fn=cmd_benchmark)
    a = ap.parse_args(argv)
    a.home = Path(a.home) if a.home else runtime_home()
    a.home.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return a.fn(a)
    except (cfgmod.ConfigError, KeyError, ValueError) as e:
        print(f"tts: {e}", file=sys.stderr)
        return 1
