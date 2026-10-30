"""A model worker: one model per process, JSON lines over stdin/stdout.

Started by pool.WorkerHandle with the model's own interpreter. Isolation is
the point: a segfault, a hang or an out-of-memory in one model kills only
this process; the parent sees a crash or a timeout and moves to the next
model. Library chatter printed to stdout is redirected to stderr so it can
never corrupt the protocol.

Requests:  {"op": "load", "device": "cpu"}
           {"op": "synth", "text", "lang", "speaker", "emotion", "draw", "out"}
           {"op": "read", "wav", "lang"}
           {"op": "health"} | {"op": "ping"} | {"op": "shutdown"}
Replies:   {"ok": true, ...} or {"ok": false, "error": "oom"|"corrupt"|"unavailable"|"exception", ...}
"""
from __future__ import annotations

import gc
import json
import os
import socket
import sys
import time
import traceback

SPEC_ENV = "TTS_WORKER_SPEC"
DENY_NETWORK_ENV = "TTS_RUNTIME_DENY_NETWORK"


def _deny_network() -> None:
    """Any connection off this machine raises. Proves rendering is offline."""
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and (host.startswith("127.") or host in ("::1", "localhost")):
            return real_connect(self, address)
        raise OSError(f"network access denied in TTS worker (tried {address}); rendering must be offline")

    socket.socket.connect = guarded
    real_getaddrinfo = socket.getaddrinfo

    def guarded_getaddrinfo(host, *a, **k):
        if host in (None, "localhost") or (isinstance(host, str) and (host.startswith("127.") or host == "::1")):
            return real_getaddrinfo(host, *a, **k)
        raise OSError(f"network access denied in TTS worker (DNS lookup for {host})")

    socket.getaddrinfo = guarded_getaddrinfo


def _is_oom(e: BaseException) -> bool:
    if isinstance(e, MemoryError):
        return True
    name = type(e).__name__
    msg = str(e).lower()
    return name == "OutOfMemoryError" or "out of memory" in msg or "cuda error: out of memory" in msg


def _free_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def awake_clock() -> float:
    """Seconds on a clock that stops while the machine sleeps. Wall time does
    not: a render that spanned a lid-close measured 3958 s of wall time. On
    Windows that clock is QueryUnbiasedInterruptTime; on Linux, monotonic."""
    if os.name == "nt":
        import ctypes

        t = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(t)):
            return t.value / 1e7          # 100 ns units
    return time.monotonic()


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> int:
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def reply(msg: dict) -> None:
        proto.write(json.dumps(msg) + "\n")
        proto.flush()

    spec = json.loads(os.environ[SPEC_ENV])
    if os.environ.get(DENY_NETWORK_ENV) == "1":
        _deny_network()

    from app.tts_runtime import licenses
    from app.tts_runtime.adapters import adapter_class, variant_of
    from app.tts_runtime.adapters.base import ModelCorrupt, Unavailable

    model_id = spec["model_id"]
    cls = adapter_class(model_id)
    try:
        licenses.check(cls.name, cls.repo, cls.license, allow_test_only=spec.get("allow_test_models", False))
    except licenses.LicenseError as e:
        reply({"ok": False, "error": "license", "detail": str(e)})
        return 2
    adapter = cls(variant_of(model_id), spec.get("options") or {})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        req = json.loads(raw)
        op = req.get("op")
        t0, a0 = time.time(), awake_clock()
        try:
            if op == "load":
                info = adapter.load(_pick_device(req.get("device", "auto")))
                reply({"ok": True, "info": info, "device": adapter.device, "secs": time.time() - t0,
                       "rss_mb": _rss_mb(), "vram_mb": _vram_mb()})
            elif op == "synth":
                import soundfile as sf

                from app.tts_runtime.fsutil import atomic_write

                out = adapter.synth(req["text"], req["lang"], req.get("speaker"), req.get("emotion"),
                                    int(req.get("draw", 0)))
                import numpy as np

                x = np.asarray(out.samples, dtype=np.float32).reshape(-1)

                def _w(f, x=x, sr=out.sr):
                    sf.write(f, x, sr, subtype="FLOAT", format="WAV")

                atomic_write(req["out"], _w)
                wall, awake = time.time() - t0, awake_clock() - a0
                reply({"ok": True, "sr": out.sr, "n": int(len(x)), "meta": out.meta,
                       "secs": wall, "awake_s": awake, "suspended_s": max(0.0, wall - awake),
                       "rss_mb": _rss_mb(), "vram_mb": _vram_mb()})
            elif op == "read":
                import soundfile as sf

                x, sr = sf.read(req["wav"], dtype="float32")
                reply({"ok": True, "text": adapter.read(x, sr, req["lang"]), "secs": time.time() - t0})
            elif op == "health":
                reply({"ok": True, "health": adapter.health_check(), "secs": time.time() - t0})
            elif op == "ping":
                reply({"ok": True})
            elif op == "shutdown":
                adapter.unload()
                reply({"ok": True})
                return 0
            else:
                reply({"ok": False, "error": "exception", "detail": f"unknown op {op!r}"})
        except ModelCorrupt as e:
            reply({"ok": False, "error": "corrupt", "path": e.path, "detail": str(e)})
        except Unavailable as e:
            reply({"ok": False, "error": "unavailable", "detail": str(e)})
        except BaseException as e:  # noqa: BLE001 -- every failure becomes a reply, not a dead worker
            if isinstance(e, KeyboardInterrupt | SystemExit):
                raise
            if _is_oom(e):
                _free_memory()
                reply({"ok": False, "error": "oom", "detail": str(e)[:500]})
            else:
                reply({"ok": False, "error": "exception", "type": type(e).__name__,
                       "detail": f"{e}"[:1000], "trace": traceback.format_exc()[-2000:]})
    return 0


def _vram_mb() -> float | None:
    """Peak CUDA memory allocated by this process, MB; None on CPU."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def _rss_mb() -> float | None:
    """Peak resident memory of this process, MB (no psutil dependency)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return None
            return round(c.PeakWorkingSetSize / 2**20, 1)
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    sys.exit(main())
