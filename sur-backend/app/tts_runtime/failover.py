"""Render one line: walk the language's chain until a model passes every check.

For each model in order:
  unsupported / unavailable / breaker open  -> skipped, logged, next model
  SYSPIN and the line has a known-bad word  -> skipped, logged, next model
  otherwise: synthesize (a retry uses a fresh draw), run the triggers
  pass -> done; fail -> logged as a switch, next model

If nothing passes: a SYSPIN render skipped only for a known-bad word is
tried as a last resort (so the line is never left silent when SYSPIN is the
only voice installed), and the best attempt is kept and FLAGGED for review.
Nothing in here raises for a bad line -- a line can never stop a batch.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.tts_runtime import knownbad
from app.tts_runtime.adapters import adapter_class
from app.tts_runtime.checks import CheckResult, reader_checks, signal_checks
from app.tts_runtime.fsutil import sha256_file
from app.tts_runtime.pool import WorkerCorrupt, WorkerError, WorkerUnavailable

logger = logging.getLogger(__name__)


def line_key(text: str, lang: str, speaker, emotion, model: str, version: str) -> str:
    return hashlib.sha256(json.dumps([text, lang, speaker, emotion, model, version],
                                     ensure_ascii=False).encode()).hexdigest()


@dataclass
class Try:
    model: str
    version: str
    key: str
    try_no: int
    outcome: str                      # ok | failed | cache_hit | timeout | crash | oom | exception | corrupt | unavailable
    check: CheckResult | None = None
    raw_path: str | None = None       # raw model output (float wav), when there is audio
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome in ("ok", "cache_hit")

    @property
    def has_audio(self) -> bool:
        return bool(self.raw_path) and Path(self.raw_path).exists()

    def severity(self) -> float:
        return self.check.severity() if self.check else 1000.0


@dataclass
class LineResult:
    state: str                         # DONE | FLAGGED
    chosen: Try | None
    tries: list[Try] = field(default_factory=list)
    reason: str = ""
    unverified: bool = False


class Context:
    """Everything a line needs; one per run."""

    def __init__(self, cfg, store, pool, breaker, knownbad, calibration, work_dir: Path, options_for,
                 job_id: str = ""):
        self.cfg = cfg
        self.store = store
        self.pool = pool
        self.breaker = breaker
        self.knownbad = knownbad
        self.calibration = calibration
        self.work_dir = Path(work_dir)
        self.options_for = options_for
        self.job_id = job_id
        self._avail: dict[str, tuple[bool, str]] = {}
        self.health_checked: set[tuple[str, str]] = set()
        self.on_corrupt = None        # callback(model, path): queue a verified re-fetch

    def available(self, model: str) -> tuple[bool, str]:
        if model not in self._avail:
            self._avail[model] = adapter_class(model).availability(self.options_for(model))
            if not self._avail[model][0]:
                self.store.event("model_unavailable", self.job_id, model=model, detail=self._avail[model][1])
                logger.warning("model %s unavailable: %s", model, self._avail[model][1])
        return self._avail[model]

    def mark_unavailable(self, model: str, why: str) -> None:
        self._avail[model] = (False, why)
        self.store.event("model_unavailable", self.job_id, model=model, detail=why)
        logger.error("model %s marked unavailable for this run: %s", model, why)

    def ensure_health(self, model: str, lang: str) -> None:
        """Health-check a model once per run, before its first line; a failure
        opens its breaker for this language."""
        if (model, lang) in self.health_checked or self.breaker.state(model, lang) == "open":
            return
        self.health_checked.add((model, lang))
        h = self.pool.health(model)
        self.store.event("health_check", self.job_id, model=model, lang=lang, detail=h)
        if h.get("ok"):
            return
        if h.get("kind") in ("unavailable", "corrupt"):
            # Refused before it could run (licence, missing interpreter, bad
            # file): not a flaky model, so not the breaker's business.
            self.mark_unavailable(model, str(h.get("detail")))
            if h.get("kind") == "corrupt" and h.get("path"):
                self.store.event("model_corrupt", self.job_id, model=model, detail={"path": h["path"]})
                if self.on_corrupt is not None:
                    self.on_corrupt(model, h["path"])
        else:
            self.breaker.health_failed(model, lang, str(h.get("detail", "failed")))

    def reset_availability(self, model: str | None = None) -> None:
        if model is None:
            self._avail.clear()
        else:
            self._avail.pop(model, None)


def _check(ctx: Context, t: Try, text: str, lang: str) -> CheckResult:
    import soundfile as sf

    assert t.raw_path is not None
    x, sr = sf.read(t.raw_path, dtype="float32")
    res = signal_checks(x, sr, text, lang, ctx.calibration)
    readers = ctx.cfg.readers.get(lang, ())
    if res.passed and readers:
        from app.tts_runtime.audio import as_mono_float, resample

        wav16 = Path(t.raw_path).with_suffix(".16k.wav")
        sf.write(str(wav16), resample(as_mono_float(x), sr, 16000), 16000, subtype="FLOAT")
        hyps = {}
        for r in readers:
            try:
                hyps[r] = ctx.pool.read(r, str(wav16), lang)
            except WorkerError as e:
                res.scores[f"reader_error_{r}"] = f"{e.kind}: {e.detail}"[:300]
                ctx.store.event("reader_error", ctx.job_id, model=r, lang=lang, detail=f"{e.kind}: {e.detail}"[:500])
                logger.error("reader %s failed (%s); line checked on signal only", r, e.kind)
        res = reader_checks(res, text, hyps)
        if not hyps:
            res.scores["unverified"] = True
    return res


def try_model(ctx: Context, line: dict, model: str, *, set_state=True, last_resort=False) -> list[Try]:
    """All tries of one model on one line (1 + retries). Records attempts, breaker, known-bad."""
    job, idx, text, lang = ctx.job_id, line["idx"], line["text"], line["lang"]
    speaker, emotion = line.get("speaker"), line.get("emotion")
    cls = adapter_class(model)
    opts = ctx.options_for(model)
    version = cls.version_for(lang, speaker, opts)
    key = line_key(text, lang, speaker, emotion, model, version)
    cached = ctx.store.cached_render(key)
    if cached and Path(cached["path"]).exists() and sha256_file(cached["path"]) == cached["sha256"]:
        t = Try(model, version, key, 0, "cache_hit", CheckResult(True, scores=json.loads(cached["scores"] or "{}")),
                cached["path"], "reused a verified render with the same key")
        ctx.store.record_skip(job, idx, model, "cache_hit", key)
        return [t]
    tries: list[Try] = []
    passed = False
    for try_no in range(1 + max(0, ctx.cfg.retries_per_model)):
        if try_no:
            ctx.store.event("retry", job, model=model, lang=lang, idx=idx,
                            detail={"try": try_no, "previous": tries[-1].outcome,
                                    "triggers": tries[-1].check.triggers if tries[-1].check else []})
            logger.warning("line %s: retrying %s (try %d) after %s", idx, model, try_no, tries[-1].outcome)
        if set_state:
            ctx.store.set_line(job, idx, "RENDERING", model=model, model_version=version)
        aid = ctx.store.start_attempt(job, idx, model, version, key, try_no)
        raw = ctx.work_dir / "attempts" / f"{key[:16]}-t{try_no}.wav"
        try:
            r = ctx.pool.synth(model, text, lang, speaker, emotion, try_no, str(raw))
            if (r.get("suspended_s") or 0) > 5:
                ctx.store.event("system_suspended", job, model=model, idx=idx,
                                detail={"suspended_s": round(r["suspended_s"]), "awake_s": round(r.get("awake_s", 0), 1)})
        except WorkerError as e:
            t = Try(model, version, key, try_no, e.kind, None, None, e.detail[:500])
            tries.append(t)
            ctx.store.finish_attempt(aid, e.kind, [e.kind], e.detail)
            logger.warning("line %s: %s %s: %s", idx, model, e.kind, e.detail[:200])
            if isinstance(e, WorkerCorrupt):
                ctx.store.event("model_corrupt", job, model=model, detail={"path": e.path, "detail": e.detail})
                ctx.mark_unavailable(model, f"corrupt model file {e.path} (re-fetch queued)")
                if ctx.on_corrupt is not None:
                    ctx.on_corrupt(model, e.path)
                return tries          # never retried, never loaded
            if isinstance(e, WorkerUnavailable):
                ctx.mark_unavailable(model, e.detail)
                return tries
            # A crash, hang or OOM says nothing about how a word is pronounced:
            # it counts against the model's breaker, never toward known-bad.
            continue
        if set_state:
            ctx.store.set_line(job, idx, "CHECKING", model=model, model_version=version)
        check = _check(ctx, Try(model, version, key, try_no, "", None, str(raw)), text, lang)
        outcome = "ok" if check.passed else "failed"
        t = Try(model, version, key, try_no, outcome, check, str(raw))
        tries.append(t)
        ctx.store.finish_attempt(aid, outcome, check.triggers, "; ".join(f["detail"] for f in check.failures),
                                 check.scores, str(raw))
        learned = ctx.knownbad.record(model, lang, text, check.passed)
        if learned:
            logger.warning("line %s: %r is now known-bad for %s", idx, learned, lang)
        if check.passed:
            ctx.store.put_render(key, model, version, str(raw), sha256_file(raw), True, check.scores)
            passed = True
            break
    if tries and not last_resort:
        ctx.breaker.record(model, lang, passed)
    return tries


def render_line(ctx: Context, line: dict) -> LineResult:
    job, idx, text, lang = ctx.job_id, line["idx"], line["text"], line["lang"]
    chain = ctx.cfg.chain_for(lang)
    all_tries: list[Try] = []
    kb_skipped: list[str] = []
    reasons: list[str] = []

    def switch(frm: str, why: str, pos: int):
        nxt = chain[pos + 1] if pos + 1 < len(chain) else None
        ctx.store.event("switch", job, model=frm, lang=lang, idx=idx, detail={"from": frm, "to": nxt, "reason": why})
        reasons.append(f"{frm}: {why}")

    for pos, model in enumerate(chain):
        cls = adapter_class(model)
        if not cls.supports(lang):
            ctx.store.record_skip(job, idx, model, "unsupported", f"{model} does not speak {lang}")
            switch(model, "unsupported language", pos)
            continue
        ok, why = ctx.available(model)
        if not ok:
            ctx.store.record_skip(job, idx, model, "unavailable", why)
            switch(model, f"unavailable ({why})", pos)
            continue
        ctx.ensure_health(model, lang)
        ok, why = ctx.available(model)
        if not ok:
            ctx.store.record_skip(job, idx, model, "unavailable", why)
            switch(model, f"unavailable ({why})", pos)
            continue
        if not ctx.breaker.allows(model, lang):
            ctx.store.record_skip(job, idx, model, "breaker_open", "circuit breaker open")
            switch(model, "circuit breaker open", pos)
            continue
        if knownbad.applies_to(model):
            bad = ctx.knownbad.hits_in(lang, text)
            if bad:
                ctx.store.record_skip(job, idx, model, "known_bad", ",".join(bad))
                kb_skipped.append(model)
                switch(model, f"known-bad word(s) {bad}", pos)
                continue
        tries = try_model(ctx, line, model)
        all_tries.extend(tries)
        if tries and tries[-1].passed:
            return LineResult("DONE", tries[-1], all_tries,
                              unverified=bool(tries[-1].check and tries[-1].check.scores.get("unverified")))
        last = tries[-1] if tries else None
        why = (last.check.triggers if last and last.check else [last.outcome if last else "no attempt"])
        switch(model, f"failed: {why}", pos)

    for model in kb_skipped:
        ctx.store.event("known_bad_last_resort", job, model=model, lang=lang, idx=idx,
                        detail="no other model passed; rendering with SYSPIN anyway and flagging for review")
        tries = try_model(ctx, line, model, last_resort=True)
        all_tries.extend(tries)
        with_audio = [t for t in tries if t.has_audio]
        if with_audio:
            best = min(with_audio, key=lambda t: t.severity())
            return LineResult("FLAGGED", best, all_tries,
                              f"contains known-bad word(s) {ctx.knownbad.hits_in(lang, text)}; no other model "
                              f"available or passing ({'; '.join(reasons)})")

    with_audio = [t for t in all_tries if t.has_audio]
    best = min(with_audio, key=lambda t: t.severity()) if with_audio else None
    reason = "every model failed: " + "; ".join(reasons) if reasons else "no model in the chain for " + lang
    return LineResult("FLAGGED", best, all_tries, reason)


def audio_of(t: Try) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(t.raw_path, dtype="float32")
    return x, sr
