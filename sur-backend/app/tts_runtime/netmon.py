"""Is the network there? Probed periodically; every change is logged.

offline_mode: auto  -- probe (default every 10 s)
              force -- always report down (network tasks stay paused)
              off   -- always report up (no probing)

Rendering never consults this: after setup it needs no network at all.
Only network tasks (downloads) do.
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

PROBE_URL_ENV = "TTS_RUNTIME_PROBE_URL"
DEFAULT_PROBE_URL = "https://huggingface.co"


def http_probe(url: str | None = None, timeout: float = 5.0) -> bool:
    import requests

    try:
        requests.head(url or os.environ.get(PROBE_URL_ENV, DEFAULT_PROBE_URL), timeout=timeout,
                      allow_redirects=False)
        return True
    except requests.RequestException:
        return False


class ConnectivityMonitor:
    def __init__(self, mode: str = "auto", probe=None, interval_s: float = 10.0, on_change=None):
        if mode not in ("auto", "force", "off"):
            raise ValueError(f"offline mode {mode!r}")
        self.mode = mode
        self.probe = probe or http_probe
        self.interval_s = interval_s
        self.on_change = on_change
        self._up: bool | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_checked = 0.0

    @property
    def state(self) -> str:
        return "unknown" if self._up is None else ("up" if self._up else "down")

    def is_up(self) -> bool:
        if self.mode == "force":
            return False
        if self.mode == "off":
            return True
        if self._up is None:
            self.check_now()
        return bool(self._up)

    def check_now(self) -> bool:
        if self.mode != "auto":
            return self.is_up()
        up = bool(self.probe())
        self.last_checked = time.time()
        if up != self._up:
            prev, self._up = self._up, up
            if prev is not None or not up:
                logger.warning("network %s", "UP (reconnected)" if up else "DOWN")
            if self.on_change:
                self.on_change(up, prev)
        return up

    def start(self) -> None:
        if self.mode != "auto" or self._thread:
            return
        self._thread = threading.Thread(target=self._loop, name="netmon", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.check_now()
            except Exception:  # noqa: BLE001 -- a probe bug must not kill the monitor
                logger.exception("connectivity probe failed")

    def stop(self) -> None:
        self._stop.set()
