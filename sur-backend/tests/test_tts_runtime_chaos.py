"""Chaos tests: cut the network, kill -9 the runner, run out of disk, corrupt a
model file, interrupt with a signal -- and check nothing is lost, duplicated
or corrupted, and that recovery needs no human step.

A local HTTP server with Range support and injectable faults stands in for
Hugging Face (a real 1.27 GB download was also killed mid-file and resumed:
see docs/tts-runtime.md). Kill and signal tests drive the real `tts` CLI in
subprocesses.
"""
from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.tts_runtime import config
from app.tts_runtime.checks import Calibration
from app.tts_runtime.config import BACKEND_DIR
from app.tts_runtime.downloader import AuthRequired, NetworkDown, download
from app.tts_runtime.fsutil import DiskFull, atomic_write, sha256_file
from app.tts_runtime.manifest import FilePin, HashMismatch, git_blob_sha1_of, lfs_pointer, read_lock
from app.tts_runtime.netmon import ConnectivityMonitor
from app.tts_runtime.netqueue import TaskQueue
from app.tts_runtime.provision import download_handler, enqueue_model
from app.tts_runtime.runner import Runner
from app.tts_runtime.store import Store

CAL = Calibration({}, {"short": {"cps_lo": 4, "cps_hi": 30}, "long": {"cps_lo": 4, "cps_hi": 30}}, 1.0, 1.0)
CAL_JSON = {"ranges": {}, "default": {"short": {"cps_lo": 4, "cps_hi": 30}, "long": {"cps_lo": 4, "cps_hi": 30}},
            "silence": {"max_lead_s": 1.0, "max_trail_s": 1.0}, "clip": {"level": 0.999, "max_fraction": 0.001}}


# --- a faulty file server -------------------------------------------------------------
class FileServer:
    def __init__(self, files: dict[str, bytes]):
        self.files = dict(files)
        self.requests: list[tuple[str, str | None]] = []
        self.drop_after: int | None = None       # bytes of body to send before cutting the connection (once)
        self.ignore_range = False
        self.wrong_once: set[str] = set()
        self.status_for: dict[str, int] = {}
        server = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                path = self.path.lstrip("/")
                rng = self.headers.get("Range")
                server.requests.append((path, rng))
                if path in server.status_for:
                    self.send_error(server.status_for[path])
                    return
                data = server.files[path]
                if path in server.wrong_once:
                    server.wrong_once.discard(path)
                    data = bytes(b ^ 0xFF for b in data)
                start = 0
                if rng and not server.ignore_range:
                    start = int(rng.split("=")[1].split("-")[0])
                body = data[start:]
                self.send_response(206 if start else 200)
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if server.drop_after is not None:
                    cut, server.drop_after = server.drop_after, None
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.close_connection = True
                    return
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def pin_for(data: bytes, path="model.bin", lfs=True) -> FilePin:
    oid = git_blob_sha1_of(lfs_pointer(hashlib.sha256(data).hexdigest(), len(data))) if lfs else git_blob_sha1_of(data)
    return FilePin(path, len(data), oid, lfs)


@pytest.fixture
def blob():
    return os.urandom(3 * 1024 * 1024 + 123)


@pytest.fixture
def server(blob):
    s = FileServer({"model.bin": blob, "config.json": b'{"x": 1}\n'})
    yield s
    s.close()


# --- network cut mid-download -----------------------------------------------------------
def test_network_cut_mid_download_resumes_from_the_partial_file(tmp_path, server, blob):
    server.drop_after = 1_000_000
    dest = tmp_path / "model.bin"
    with pytest.raises(NetworkDown, match="kept model.bin.partial"):
        download(f"{server.url}/model.bin", dest, pin_for(blob))
    part = tmp_path / "model.bin.partial"
    kept = part.stat().st_size
    assert 0 < kept < len(blob) and not dest.exists()
    sha, stats = download(f"{server.url}/model.bin", dest, pin_for(blob))
    assert server.requests[-1] == ("model.bin", f"bytes={kept}-"), "resumed with a Range request"
    assert stats["resumed_from"] == kept and stats["fetched"] == len(blob) - kept and not stats["restarted"]
    assert sha == hashlib.sha256(blob).hexdigest() and dest.read_bytes() == blob and not part.exists()


def test_a_server_that_ignores_range_restarts_and_says_so(tmp_path, server, blob):
    server.drop_after = 500_000
    with pytest.raises(NetworkDown):
        download(f"{server.url}/model.bin", tmp_path / "model.bin", pin_for(blob))
    server.ignore_range = True
    sha, stats = download(f"{server.url}/model.bin", tmp_path / "model.bin", pin_for(blob))
    assert stats["restarted"] is True and sha == hashlib.sha256(blob).hexdigest()


def test_a_download_that_fails_its_hash_is_deleted_and_never_used(tmp_path, server, blob):
    server.wrong_once.add("model.bin")
    dest = tmp_path / "model.bin"
    with pytest.raises(HashMismatch, match="refusing to use it"):
        download(f"{server.url}/model.bin", dest, pin_for(blob))
    assert not dest.exists() and not (tmp_path / "model.bin.partial").exists()
    sha, _ = download(f"{server.url}/model.bin", dest, pin_for(blob))        # re-fetched, clean
    assert sha == hashlib.sha256(blob).hexdigest()


def test_a_gated_file_without_a_token_fails_once_and_is_not_retried(tmp_path, server):
    server.status_for["config.json"] = 401
    with pytest.raises(AuthRequired, match="HF_TOKEN"):
        download(f"{server.url}/config.json", tmp_path / "c.json", pin_for(b'{"x": 1}\n', "config.json", False))


def _write_pin(pins_dir: Path, server, files: dict[str, bytes], name="fakemodel") -> None:
    pins_dir.mkdir(parents=True, exist_ok=True)
    entries = [{"path": p, "size": len(d), "git_oid": pin_for(d, p, p.endswith(".bin")).git_oid, "lfs": p.endswith(".bin")}
               for p, d in files.items()]
    (pins_dir / f"{name}.json").write_text(json.dumps(
        {"name": name, "repo": "local/fake-model", "revision": "0" * 40, "license": "LicenseRef-no-weights",
         "gated": False, "base_url": server.url, "files": entries}), encoding="utf-8")


def test_task_queue_pauses_offline_and_resumes_on_reconnect_with_backoff(tmp_path, server, blob):
    pins = tmp_path / "pins"
    _write_pin(pins, server, {"model.bin": blob, "config.json": b'{"x": 1}\n'})
    store, net = Store(tmp_path / "s.sqlite3"), {"up": False}
    mon = ConnectivityMonitor("auto", probe=lambda: net["up"])
    opts = {"behavior": {"pins": ["fakemodel"], "pins_dir": str(pins)}}
    assert enqueue_model(store, tmp_path, "fake:a", opts)["queued"] == 2
    q = TaskQueue(store, mon, (0.05, 0.1), {"download": download_handler(tmp_path)})
    q.run_once()
    assert {t["state"] for t in store.tasks()} == {"PAUSED"}
    net["up"] = True
    mon.check_now()
    for _ in range(50):
        q.run_once()
        if not q.pending():
            break
        time.sleep(0.05)
    assert {t["state"] for t in store.tasks()} == {"DONE"}
    kinds = [e["kind"] for e in store.events()]
    assert kinds.count("pause") == 2 and kinds.count("resume") == 2 and kinds.count("task_done") == 2
    assert read_lock(tmp_path / "models" / "fakemodel" / ("0" * 40))["sha256"]["model.bin"] == hashlib.sha256(blob).hexdigest()


def test_a_download_killed_while_running_is_requeued_and_resumed(tmp_path, server, blob):
    """The bug a real kill -9 of a 1.27 GB download exposed: the task stayed RUNNING forever."""
    pins = tmp_path / "pins"
    _write_pin(pins, server, {"model.bin": blob})
    store = Store(tmp_path / "s.sqlite3")
    enqueue_model(store, tmp_path, "fake:a", {"behavior": {"pins": ["fakemodel"], "pins_dir": str(pins)}})
    tid = store.tasks()[0]["id"]
    store.set_task(tid, "RUNNING")                      # as left by a process that died mid-download
    q = TaskQueue(store, ConnectivityMonitor("off"), (0.05,), {"download": download_handler(tmp_path)})
    while q.pending():
        q.run_once()
    assert store.tasks()[0]["state"] == "DONE" and store.events(None, "task_recovered")


# --- network cut mid-batch -----------------------------------------------------------------
def make_runner(tmp_path, chains, options=None, netmon=None, **kw):
    cfg = config.parse({"chains": {**chains, "default": chains[next(iter(chains))]}, "offline_mode": "auto",
                        "retries": {"per_model": 0, "network_backoff": [0.1, 0.2]}, "scene_consistency": False},
                       allow_test_models=True, source="test")
    cfg = dataclasses.replace(cfg, synth_timeout_s=20)
    return Runner(cfg, tmp_path / "home", calibration=CAL, options=options or {}, netmon=netmon,
                  known_bad_dir=tmp_path / "kb", heartbeat_s=0.3, **kw)


def test_network_cut_mid_batch_rendering_continues_network_tasks_pause_then_resume(tmp_path, server, blob):
    pins = tmp_path / "pins"
    _write_pin(pins, server, {"model.bin": blob})
    net = {"up": False}
    mon = ConnectivityMonitor("auto", probe=lambda: net["up"], interval_s=0.1)
    r = make_runner(tmp_path, {"hi": ["fake:a"]}, {"fake:a": {"behavior": {"sleep_s": 0.6}}}, netmon=mon)
    enqueue_model(r.store, r.home, "fake:b", {"behavior": {"pins": ["fakemodel"], "pins_dir": str(pins)}})
    jid = r.submit({"lines": [{"text": f"पंक्ति {i}", "lang": "hi"} for i in range(8)]})
    out = {}
    t = threading.Thread(target=lambda: out.update(r.run(jid)))
    t.start()
    deadline = time.time() + 60
    while r.store.counts(jid)["DONE"] < 3 and time.time() < deadline:
        time.sleep(0.1)
    done_while_offline = r.store.counts(jid)["DONE"]
    assert done_while_offline >= 3, "rendering continued with the network down"
    assert r.store.tasks()[0]["state"] == "PAUSED"
    net["up"] = True                                    # the network comes back
    t.join(120)
    for _ in range(100):
        if r.store.tasks()[0]["state"] == "DONE" or not t.is_alive():
            break
        time.sleep(0.1)
    assert out["status"] == "DONE" and out["counts"]["DONE"] == 8
    kinds = [e["kind"] for e in r.store.events()]
    assert "pause" in kinds and "resume" in kinds
    assert r.store.tasks()[0]["state"] == "DONE", "the paused download resumed by itself"


# --- kill -9, signals, supervisor ---------------------------------------------------------------
def _cli_env(tmp_path, options: dict) -> dict:
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(CAL_JSON))
    env = dict(os.environ, PYTHONPATH=str(BACKEND_DIR), PYTHONUTF8="1", TTS_RUNTIME_ALLOW_TEST_MODELS="1",
               TTS_RUNTIME_MODEL_OPTIONS=json.dumps(options), TTS_RUNTIME_CALIBRATION=str(cal))
    return env


def _cli_files(tmp_path, n=8):
    cfgp = tmp_path / "chains.yaml"
    cfgp.write_text("chains: {hi: ['fake:a'], default: ['fake:a']}\noffline_mode: off\nscene_consistency: false\n"
                    "retries: {per_model: 0}\n", encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"job_id": "chaos", "lines": [{"text": f"पंक्ति {i}", "lang": "hi"} for i in range(n)]}),
                    encoding="utf-8")
    return cfgp, spec


def _db_counts(home: Path) -> dict:
    try:
        con = sqlite3.connect(f"file:{home / 'runtime.sqlite3'}?mode=ro", uri=True, timeout=5)
        rows = con.execute("SELECT state, COUNT(*) FROM lines WHERE job_id='chaos' GROUP BY state").fetchall()
        con.close()
        return dict(rows)
    except sqlite3.Error:
        return {}


def _wait(pred, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.2)
    return False


def _cli(*args, home):
    return [sys.executable, "-m", "app.tts_runtime", "--home", str(home), *args]


def test_kill_9_mid_render_then_resume_finishes_with_no_duplicates_or_corrupt_files(tmp_path):
    home = tmp_path / "home"
    cfgp, spec = _cli_files(tmp_path)
    env = _cli_env(tmp_path, {"fake:a": {"behavior": {"sleep_s": 1.0}}})
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
    p = subprocess.Popen(_cli("--config", str(cfgp), "run", str(spec), home=home), cwd=BACKEND_DIR, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    assert _wait(lambda: _db_counts(home).get("DONE", 0) >= 3 and _db_counts(home).get("RENDERING", 0) == 1)
    p.kill()                                                   # kill -9: no handler runs
    p.wait(30)
    before = _db_counts(home)
    assert before.get("RENDERING") == 1, before                # a line really was in flight

    s = Store(home / "runtime.sqlite3")
    done_before = {ln["idx"] for ln in s.lines("chaos", ("DONE",))}
    r = subprocess.run(_cli("--config", str(cfgp), "resume", "chaos", home=home), cwd=BACKEND_DIR, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    s = Store(home / "runtime.sqlite3")
    lines = s.lines("chaos")
    assert [ln["state"] for ln in lines] == ["DONE"] * 8
    outs = sorted((home / "jobs" / "chaos").glob("*.wav"))
    assert [o.name for o in outs] == [f"{i:05d}.wav" for i in range(8)], "exactly one output per line"
    for ln in lines:
        assert sha256_file(ln["output_path"]) == ln["output_sha256"]
    assert not list(home.rglob(".tmp-*")), "no temp files left"
    for idx in done_before:                                    # finished lines were not rendered again
        oks = [a for a in s.attempts("chaos", idx) if a["outcome"] in ("ok", "cache_hit")]
        assert len(oks) == 1, (idx, [a["outcome"] for a in s.attempts("chaos", idx)])
    assert any(a["outcome"] == "interrupted" for a in s.attempts("chaos"))
    assert s.events("chaos", "restart_recovered")


@pytest.mark.skipif(os.name != "nt" and not hasattr(signal, "SIGINT"), reason="needs signals")
def test_a_stop_signal_finishes_the_current_line_persists_and_exits_cleanly(tmp_path):
    home = tmp_path / "home"
    cfgp, spec = _cli_files(tmp_path)
    env = _cli_env(tmp_path, {"fake:a": {"behavior": {"sleep_s": 1.0}}})
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
    p = subprocess.Popen(_cli("--config", str(cfgp), "run", str(spec), home=home), cwd=BACKEND_DIR, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kw)
    assert _wait(lambda: _db_counts(home).get("DONE", 0) >= 2)
    p.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    assert p.wait(60) == 3, p.stderr.read()[-2000:]            # 3 = interrupted, resume to continue
    c = _db_counts(home)
    assert c.get("RENDERING", 0) == 0 and c.get("CHECKING", 0) == 0, c    # the current line was finished
    assert Store(home / "runtime.sqlite3").job("chaos")["status"] == "INTERRUPTED"
    r = subprocess.run(_cli("--config", str(cfgp), "resume", "chaos", home=home), cwd=BACKEND_DIR, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and _db_counts(home) == {"DONE": 8}


def test_the_supervisor_restarts_a_killed_runner_and_finishes_the_job(tmp_path):
    home = tmp_path / "home"
    cfgp, spec = _cli_files(tmp_path)
    env = _cli_env(tmp_path, {"fake:a": {"behavior": {"sleep_s": 0.8}}})
    sup = subprocess.Popen(_cli("--config", str(cfgp), "supervise", str(spec), home=home), cwd=BACKEND_DIR, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hb = home / "heartbeat.json"
    assert _wait(lambda: _db_counts(home).get("DONE", 0) >= 2 and hb.exists())
    pid = json.loads(hb.read_text())["pid"]
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)   # kill -9 equivalent
    else:
        os.kill(pid, signal.SIGKILL)
    assert sup.wait(300) == 0
    s = Store(home / "runtime.sqlite3")
    assert _db_counts(home) == {"DONE": 8}
    assert s.events("chaos", "supervisor_restart"), "the supervisor restarted the runner"


def test_resume_re_renders_a_done_line_whose_output_is_missing_or_changed(tmp_path):
    r = make_runner(tmp_path, {"hi": ["fake:a"]}, netmon=ConnectivityMonitor("off"))
    jid = r.submit({"lines": [{"text": f"पंक्ति {i}", "lang": "hi"} for i in range(3)]})
    r.run(jid)
    lines = r.store.lines(jid)
    Path(lines[0]["output_path"]).unlink()                              # deleted
    Path(lines[1]["output_path"]).write_bytes(b"RIFF not the audio")    # tampered
    rec = r.recover(jid)
    assert sorted(rec["invalid_outputs"]) == [0, 1]
    assert r.store.line(jid, 2)["state"] == "DONE", "the intact line is left alone"
    res = r.run(jid)
    assert res["status"] == "DONE"
    for ln in r.store.lines(jid):
        assert sha256_file(ln["output_path"]) == ln["output_sha256"]


# --- disk -----------------------------------------------------------------------------------
def test_atomic_write_on_enospc_leaves_the_old_file_and_no_temp(tmp_path):
    target = tmp_path / "out.wav"
    target.write_bytes(b"previous complete file")

    def writer(f):
        f.write(b"half")
        raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.raises(DiskFull):
        atomic_write(target, writer)
    assert target.read_bytes() == b"previous complete file"
    assert list(tmp_path.iterdir()) == [target]


def test_low_disk_pauses_the_batch_with_an_alert_then_resumes_without_touching_files(tmp_path):
    free = {"mb": 10_000.0}
    r = make_runner(tmp_path, {"hi": ["fake:a"]}, {"fake:a": {"behavior": {"sleep_s": 0.3}}},
                    netmon=ConnectivityMonitor("off"), free_mb_fn=lambda _p: free["mb"], disk_poll_s=0.1)
    jid = r.submit({"lines": [{"text": f"पंक्ति {i}", "lang": "hi"} for i in range(6)]})
    out = {}
    t = threading.Thread(target=lambda: out.update(r.run(jid)))
    t.start()
    assert _wait(lambda: r.store.counts(jid)["DONE"] >= 2, 60)
    free["mb"] = 100.0                                       # below the 2048 MB threshold
    assert _wait(lambda: r.store.job(jid)["status"] == "PAUSED", 30)
    outs = sorted((r.home / "jobs" / jid).glob("*.wav"))
    snapshot = {o.name: o.read_bytes() for o in outs}
    time.sleep(1.0)
    assert sorted((r.home / "jobs" / jid).glob("*.wav")) == outs, "nothing written while paused"
    assert {o.name: o.read_bytes() for o in outs} == snapshot, "nothing changed while paused"
    free["mb"] = 10_000.0
    t.join(120)
    assert out["status"] == "DONE" and out["counts"]["DONE"] == 6
    assert [k for k, _ in r.alerts] == ["disk_low"]
    assert r.store.events(jid, "pause") and r.store.events(jid, "resume")


def test_disk_full_while_writing_an_output_puts_the_line_back_and_finishes(tmp_path, monkeypatch):
    import app.tts_runtime.runner as runner_mod

    real = runner_mod.write_wav_atomic
    calls = {"n": 0}

    def flaky(path, audio, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise DiskFull(errno.ENOSPC, f"disk full writing {path}; nothing was replaced")
        return real(path, audio, *a, **k)

    monkeypatch.setattr(runner_mod, "write_wav_atomic", flaky)
    r = make_runner(tmp_path, {"hi": ["fake:a"]}, netmon=ConnectivityMonitor("off"))
    jid = r.submit({"lines": [{"text": f"पंक्ति {i}", "lang": "hi"} for i in range(3)]})
    res = r.run(jid)
    assert res["status"] == "DONE" and res["counts"]["DONE"] == 3
    assert r.store.events(jid, "disk_full") and ("disk_full" in [k for k, _ in r.alerts])
    assert not list(r.home.rglob(".tmp-*"))


# --- corrupted model file ------------------------------------------------------------------
def test_a_corrupted_model_file_is_caught_never_loaded_refetched_and_restored(tmp_path, server, blob):
    pins = tmp_path / "pins"
    _write_pin(pins, server, {"model.bin": blob})
    home = tmp_path / "home"
    opts_a = {"behavior": {"pins": ["fakemodel"], "pins_dir": str(pins)}}
    st = Store(home / "runtime.sqlite3")
    enqueue_model(st, home, "fake:a", opts_a)
    q = TaskQueue(st, ConnectivityMonitor("off"), (0.05,), {"download": download_handler(home)})
    while q.pending():
        q.run_once()
    model_file = home / "models" / "fakemodel" / ("0" * 40) / "model.bin"
    good_sha = sha256_file(model_file)
    data = bytearray(model_file.read_bytes())
    data[1000] ^= 0xFF                                         # one flipped byte on disk
    model_file.write_bytes(bytes(data))

    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]},
                    {"fake:a": opts_a, "fake:b": {"behavior": {"sleep_s": 1.0}}}, netmon=ConnectivityMonitor("off"))
    jid = r.submit({"lines": [{"text": f"पंक्ति {i}", "lang": "hi"} for i in range(6)]})
    res = r.run(jid)
    assert res["status"] == "DONE" and res["counts"]["DONE"] == 6
    models = [ln["model"] for ln in r.store.lines(jid)]
    assert models[0] == "fake:b", "the corrupt model never rendered a line"
    assert "fake:a" in models[1:], "restored after the verified re-fetch"
    kinds = [e["kind"] for e in r.store.events()]
    for k in ("model_corrupt", "refetch_queued", "task_done", "model_restored"):
        assert k in kinds, (k, kinds)
    assert sha256_file(model_file) == good_sha
