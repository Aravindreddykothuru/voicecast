"""`tts setup` and `tts warmup`.

setup (idempotent; safe to re-run after any interruption):
  * licence-checks every model in the chains and every reader;
  * reads HF_TOKEN from the environment (never logged). Without it the gated
    models are disabled with a warning -- not a crash -- and skipped at runtime;
  * queues every pinned file of every model as a network task: resumable
    download, verified against the pin, then recorded in the model's lock;
    files already verified are skipped;
  * makes sure SYSPIN voices and the readers are cached at their pinned
    revisions (huggingface_hub resumes its own partial downloads).

warmup: every model x language in the chains speaks one test sentence;
reports pass/fail, load time and memory (VRAM on GPU, peak RSS on CPU).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from app.tts_runtime import licenses
from app.tts_runtime.config import RuntimeConfig, adapter_name
from app.tts_runtime.manifest import load_pin, model_dir, read_lock, write_lock

logger = logging.getLogger(__name__)

# Which pins each model needs on disk.
MODEL_PINS: dict[str, tuple[str, ...]] = {
    "indic_parler": ("indic_parler", "flan_t5_large_tokenizer"),
    "indicf5": ("indicf5", "vocos"),
    "parler_tiny": ("parler_tiny",),
}
READER_REVISIONS = {
    "Harveenchadha/vakyansh-wav2vec2-telugu-tem-100": "f0b4778462800eaa70163bfee6bf97710cc28f27",
    "Harveenchadha/vakyansh-wav2vec2-kannada-knm-560": "0564782ee3f5a47f1db8af94bcdc942bb0d5bb29",
    "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200": "e2568c3f7868d8aa3aaabcf28fa100d10d54c170",
    "Harveenchadha/vakyansh-wav2vec2-marathi-mrm-100": "82f3b0fac2c26ffd8965a031a90ad61132b233ee",
    "Harveenchadha/vakyansh-wav2vec2-bengali-bnm-200": "82a1dfce91a42debc80e51f0b3f3b19e10bab633",
}


def hf_token() -> str | None:
    from app.config import hf_token as _t

    return _t()


def download_handler(home: Path):
    """Network-task handler: fetch one pinned file, verify, record in the lock."""
    from app.tts_runtime.downloader import AuthRequired, download

    def handle(payload: dict) -> None:
        pin = load_pin(payload["pin"], *([Path(payload["pins_dir"])] if payload.get("pins_dir") else []))
        f = next(x for x in pin.files if x.path == payload["file"])
        token = hf_token() if pin.gated else None
        if pin.gated and not token:
            raise AuthRequired(f"{pin.repo} is gated and HF_TOKEN is not set")
        folder = model_dir(home, pin)
        sha, stats = download(pin.url(f), folder / f.path, f, token=token)
        lock = read_lock(folder).get("sha256", {})
        lock[f.path] = sha
        write_lock(folder, pin, lock)
        logger.info("verified %s/%s (resumed from %d bytes, fetched %d)", pin.name, f.path,
                    stats["resumed_from"], stats["fetched"])

    return handle


def task_id(pin_name: str, revision: str, path: str) -> str:
    return f"dl:{pin_name}:{revision[:12]}:{path}"


def pins_for(model: str, options: dict | None = None) -> tuple[tuple[str, ...], Path | None]:
    options = options or {}
    beh = options.get("behavior") or {}
    pins = options.get("pins") or beh.get("pins") or MODEL_PINS.get(adapter_name(model), ())
    pins_dir = options.get("pins_dir") or beh.get("pins_dir")
    return tuple(pins), (Path(pins_dir) if pins_dir else None)


def _load(pin_name: str, pins_dir: Path | None):
    return load_pin(pin_name, pins_dir) if pins_dir else load_pin(pin_name)


def enqueue_model(store, home: Path, model: str, options: dict | None = None) -> dict:
    """Queue every not-yet-verified pinned file of `model`."""
    out = {"queued": 0, "verified": 0, "skipped_gated": []}
    token = hf_token()
    pins, pins_dir = pins_for(model, options)
    for pin_name in pins:
        pin = _load(pin_name, pins_dir)
        licenses.check(adapter_name(model), pin.repo, pin.license,
                       allow_test_only=adapter_name(model) in ("parler_tiny", "fake"))
        if pin.gated and not token:
            out["skipped_gated"].append(pin.repo)
            continue
        lock = read_lock(model_dir(home, pin)).get("sha256", {})
        for f in pin.files:
            if f.path in lock and (model_dir(home, pin) / f.path).exists():
                out["verified"] += 1
                continue
            tid = task_id(pin_name, pin.revision, f.path)
            row = store.q("SELECT state FROM tasks WHERE id=?", (tid,))
            if row and row[0]["state"] in ("DONE", "FAILED"):
                store.set_task(tid, "QUEUED", attempts=0, next_at=0, last_error=None)
                out["queued"] += 1
            elif store.enqueue_task(tid, "download", {"pin": pin_name, "file": f.path,
                                                       "pins_dir": str(pins_dir) if pins_dir else None}):
                out["queued"] += 1
    return out


def enqueue_refetch(store, model: str, path: str, options: dict | None = None) -> None:
    """A verified file went bad on disk: forget it and queue it again."""
    p = Path(path)
    pins, pins_dir = pins_for(model, options)
    for pin_name in pins:
        pin = _load(pin_name, pins_dir)
        for f in pin.files:
            if p.as_posix().endswith(f"{pin.name}/{pin.revision}/{f.path}"):
                folder = p.parents[len(Path(f.path).parts) - 1]
                lock = read_lock(folder).get("sha256", {})
                lock.pop(f.path, None)
                write_lock(folder, pin, lock)
                tid = task_id(pin_name, pin.revision, f.path)
                payload = {"pin": pin_name, "file": f.path, "pins_dir": str(pins_dir) if pins_dir else None}
                if not store.enqueue_task(tid, "download", payload):
                    store.set_task(tid, "QUEUED", attempts=0, next_at=0, last_error="re-fetch after corruption")
                store.event("refetch_queued", model=model, detail={"file": f.path})
                return


def ensure_hf_cached(repo: str, revision: str, files: list[str] | None = None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision, allow_patterns=files)


def setup(cfg: RuntimeConfig, home: Path, store, netmon, *, wait: bool = True, poll_s: float = 1.0,
          timeout_s: float | None = None) -> dict:
    from app.tts_runtime.netqueue import TaskQueue

    report: dict = {"models": {}, "token": bool(hf_token()), "warnings": []}
    if not report["token"]:
        msg = ("HF_TOKEN is not set: gated models (indic_parler, indicf5, IndicConformer) cannot be downloaded "
               "and will be skipped at runtime, logged as unavailable. Set HF_TOKEN (read scope, licence "
               "accepted on huggingface.co) and re-run `make setup-tts` to enable them.")
        logger.warning(msg)
        report["warnings"].append(msg)
    for model in cfg.models():
        name = adapter_name(model)
        if name == "syspin":
            from app.providers.tts.syspin_manifest import SYSPIN_MANIFEST

            try:
                for repo, pin in SYSPIN_MANIFEST.items():
                    ensure_hf_cached(repo, pin.revision, list(pin.files) + (["extra.py"] if pin.ships_extra_py else []))
                report["models"][model] = {"status": "cached"}
            except Exception as e:  # noqa: BLE001 -- offline: report, keep going
                report["models"][model] = {"status": "incomplete", "error": f"{type(e).__name__}: {e}"}
            continue
        report["models"][model] = enqueue_model(store, home, model)
    for lang, readers in cfg.readers.items():
        for r in readers:
            if r == "vakyansh":
                from app.tts_runtime.adapters.vakyansh import READER_REPOS

                repo = READER_REPOS.get(lang)
                if repo:
                    try:
                        ensure_hf_cached(repo, READER_REVISIONS[repo])
                        report["models"][f"reader:{lang}"] = {"status": "cached"}
                    except Exception as e:  # noqa: BLE001
                        report["models"][f"reader:{lang}"] = {"status": "incomplete", "error": str(e)}
    if wait:
        q = TaskQueue(store, netmon, cfg.network_backoff_s, {"download": download_handler(home)})
        t0 = time.time()
        while q.pending():
            q.run_once()
            if timeout_s is not None and time.time() - t0 > timeout_s:
                report["warnings"].append(f"setup stopped waiting after {timeout_s:.0f}s; re-run to continue")
                break
            time.sleep(poll_s)
    report["tasks"] = {r["id"]: {"state": r["state"], "attempts": r["attempts"], "error": r["last_error"]}
                       for r in store.tasks()}
    return report


WARMUP_TEXT = Path(__file__).resolve().parent / "data" / "warmup_sentences.json"


def warmup(runner, langs: list[str] | None = None) -> list[dict]:
    """Every model x language speaks one sentence. Returns one row per pair."""
    from app.tts_runtime.adapters import adapter_class
    from app.tts_runtime.checks import signal_checks
    from app.tts_runtime.pool import WorkerError

    sentences = json.loads(WARMUP_TEXT.read_text(encoding="utf-8"))
    cfg = runner.cfg
    rows = []
    pairs = []
    for lang in sorted(set(cfg.chains) | set(sentences)):
        if langs and lang not in langs:
            continue
        for model in cfg.chain_for(lang):
            pairs.append((model, lang))
    for model, lang in pairs:
        cls = adapter_class(model)
        row = {"model": model, "lang": lang}
        if not cls.supports(lang):
            rows.append({**row, "result": "unsupported"})
            continue
        ok, why = cls.availability(runner.options_for(model))
        if not ok:
            rows.append({**row, "result": "unavailable", "why": why})
            continue
        text = sentences.get(lang)
        if not text:
            rows.append({**row, "result": "no test sentence"})
            continue
        out = runner.home / "warmup" / f"{model.replace(':', '_')}-{lang}.wav"
        t0 = time.time()
        try:
            r = runner.pool.synth(model, text, lang, None, None, 0, str(out))
        except WorkerError as e:
            rows.append({**row, "result": "fail", "why": f"{e.kind}: {e.detail[:200]}", "secs": round(time.time() - t0, 1)})
            continue
        import soundfile as sf

        x, sr = sf.read(str(out), dtype="float32")
        chk = signal_checks(x, sr, text, lang, runner.calibration)
        st = runner.pool.stats.get(model, {})
        rows.append({**row, "result": "pass" if chk.passed else "fail", "why": "; ".join(f["detail"] for f in chk.failures),
                     "synth_s": round(r.get("secs", 0), 2), "load_s": st.get("load_s"), "device": st.get("device"),
                     "rss_mb": r.get("rss_mb"), "vram_mb": r.get("vram_mb"), "audio_s": round(len(x) / sr, 2)})
    return rows
