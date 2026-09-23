"""GET /health on 127.0.0.1:<port>: model status, queue depth, network state,
last error, heartbeat age. Served from a daemon thread while a job runs;
`tts status` reads the same facts from the heartbeat file and the store."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def snapshot(runner) -> dict:
    job = runner.current.get("job")
    counts = runner.store.counts(job) if job else {}
    return {
        "at": time.time(),
        "job": job, "line": runner.current.get("idx"), "state": runner.current.get("state"),
        "queue": {"pending_lines": counts.get("PENDING", 0), "lines": counts,
                  "network_tasks": {s: len(runner.store.tasks((s,))) for s in ("QUEUED", "RUNNING", "PAUSED", "FAILED")}},
        "network": runner.netmon.state,
        "models": runner.model_status(),
        "loaded": runner.pool.status(),
        "last_error": runner.last_error,
        "alerts": runner.alerts[-10:],
    }


def serve(runner, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") not in ("/health", ""):
                self.send_error(404)
                return
            body = json.dumps(snapshot(runner), default=str, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, name="health", daemon=True).start()
    return srv
