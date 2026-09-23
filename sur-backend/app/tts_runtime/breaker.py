"""Circuit breaker per (model, language), persisted in the store.

closed    -> open       when the model failed more than fail_ratio of its last
                        `window` lines (evaluated once it has min_samples), or
                        failed its health check
open      -> half_open  after cooldown_s: the next line is a trial
half_open -> closed     trial passed (window cleared)
half_open -> open       trial failed (cooldown restarts)

Every open and close is logged at WARNING, recorded as an event, and sent to
the alert hook -- which the run report and the health endpoint both show.
"""
from __future__ import annotations

import logging
import time

from app.tts_runtime.config import BreakerConfig

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(self, store, cfg: BreakerConfig, alert=None, clock=time.time):
        self.store = store
        self.cfg = cfg
        self.alert = alert or (lambda kind, msg: None)
        self.clock = clock

    def state(self, model: str, lang: str) -> str:
        row = self.store.breaker(model, lang)
        if row is None or row["state"] == "closed":
            return "closed"
        if row["state"] == "open" and self.clock() - (row["opened_at"] or 0) >= self.cfg.cooldown_s:
            self.store.set_breaker(model, lang, "half_open", "cooldown elapsed; next line is a trial",
                                   row["opened_at"])
            self._notify("breaker_half_open", model, lang, "cooldown elapsed; trying one line")
            return "half_open"
        return row["state"]

    def allows(self, model: str, lang: str) -> bool:
        return self.state(model, lang) != "open"

    def record(self, model: str, lang: str, ok: bool) -> None:
        state = self.state(model, lang)
        if state == "half_open":
            if ok:
                self.store.clear_outcomes(model, lang)
                self.store.set_breaker(model, lang, "closed", "trial line passed")
                self._notify("breaker_close", model, lang, "trial line passed")
            else:
                self._open(model, lang, "trial line failed after cooldown")
            return
        self.store.record_outcome(model, lang, ok)
        recent = self.store.recent_outcomes(model, lang, self.cfg.window)
        if len(recent) < self.cfg.min_samples:
            return
        fails = recent.count(0)
        if fails / len(recent) > self.cfg.fail_ratio:
            self._open(model, lang, f"failed {fails} of its last {len(recent)} lines "
                                    f"(> {self.cfg.fail_ratio:.0%})")

    def health_failed(self, model: str, lang: str, detail: str) -> None:
        self._open(model, lang, f"health check failed: {detail}")

    def _open(self, model: str, lang: str, reason: str) -> None:
        self.store.set_breaker(model, lang, "open", reason, self.clock())
        self._notify("breaker_open", model, lang, reason)

    def _notify(self, kind: str, model: str, lang: str, reason: str) -> None:
        msg = f"CIRCUIT BREAKER {kind.split('_', 1)[1].upper()}: {model} for {lang} -- {reason}"
        logger.warning(msg)
        self.store.event(kind, model=model, lang=lang, detail=reason)
        self.alert(kind, msg)
