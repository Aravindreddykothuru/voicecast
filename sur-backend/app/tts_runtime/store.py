"""Durable runtime state: SQLite in WAL mode, synchronous=FULL.

Each line moves PENDING -> RENDERING -> CHECKING -> DONE, or FLAGGED. Every
transition is its own committed transaction, so after a crash the database
says exactly which line was in flight; `recover()` puts it back to PENDING.
FULL synchronous mode means a committed transition survives power loss, not
just a process crash.

Also here: every attempt (the audit trail and the self-learning input), the
content-addressed render cache that makes reruns idempotent, the circuit
breaker's window, known-bad word history, run events, and the queue of
network tasks that pause while offline.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

LINE_STATES = ("PENDING", "RENDERING", "CHECKING", "DONE", "FLAGGED")
TASK_STATES = ("QUEUED", "RUNNING", "PAUSED", "DONE", "FAILED")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, created_at REAL NOT NULL, updated_at REAL NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('CREATED','RUNNING','PAUSED','INTERRUPTED','DONE')),
  spec_json TEXT NOT NULL, out_dir TEXT NOT NULL, pause_reason TEXT);
CREATE TABLE IF NOT EXISTS lines(
  job_id TEXT NOT NULL REFERENCES jobs(id), idx INTEGER NOT NULL,
  text TEXT NOT NULL, lang TEXT NOT NULL, speaker TEXT, emotion TEXT, scene TEXT,
  state TEXT NOT NULL CHECK(state IN ('PENDING','RENDERING','CHECKING','DONE','FLAGGED')),
  model TEXT, model_version TEXT, key TEXT, output_path TEXT, output_sha256 TEXT,
  flag_reason TEXT, updated_at REAL, PRIMARY KEY(job_id, idx));
CREATE TABLE IF NOT EXISTS attempts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, idx INTEGER, model TEXT, model_version TEXT,
  key TEXT, try_no INTEGER, started_at REAL, finished_at REAL, outcome TEXT, triggers TEXT,
  detail TEXT, scores TEXT, audio_path TEXT);
CREATE INDEX IF NOT EXISTS attempts_line ON attempts(job_id, idx);
CREATE TABLE IF NOT EXISTS renders(
  key TEXT PRIMARY KEY, model TEXT, model_version TEXT, path TEXT, sha256 TEXT,
  passed INTEGER, scores TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS word_history(
  id INTEGER PRIMARY KEY AUTOINCREMENT, lang TEXT, word TEXT, model TEXT, ok INTEGER, at REAL);
CREATE INDEX IF NOT EXISTS word_history_w ON word_history(lang, word, model);
CREATE TABLE IF NOT EXISTS model_outcomes(
  id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT, lang TEXT, ok INTEGER, at REAL);
CREATE INDEX IF NOT EXISTS model_outcomes_ml ON model_outcomes(model, lang);
CREATE TABLE IF NOT EXISTS breakers(
  model TEXT, lang TEXT, state TEXT CHECK(state IN ('closed','open','half_open')),
  opened_at REAL, reason TEXT, PRIMARY KEY(model, lang));
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, at REAL, kind TEXT, model TEXT, lang TEXT,
  idx INTEGER, detail TEXT);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id, kind);
CREATE TABLE IF NOT EXISTS tasks(
  id TEXT PRIMARY KEY, kind TEXT, payload TEXT,
  state TEXT CHECK(state IN ('QUEUED','RUNNING','PAUSED','DONE','FAILED')),
  attempts INTEGER DEFAULT 0, next_at REAL DEFAULT 0, last_error TEXT, updated_at REAL);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None,
                                   check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        mode = self._db.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if mode.lower() != "wal":
            raise RuntimeError(f"SQLite refused WAL mode for {self.path} (got {mode})")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def tx(self):
        """One committed transaction (BEGIN IMMEDIATE: writers serialise)."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            else:
                self._db.execute("COMMIT")

    def q(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # --- jobs & lines ------------------------------------------------------
    def create_job(self, job_id: str, spec: dict, out_dir: str, lines: list[dict]) -> bool:
        """Create the job with all its lines PENDING. False if it already exists."""
        now = time.time()
        with self.tx() as db:
            if db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
                return False
            db.execute("INSERT INTO jobs(id, created_at, updated_at, status, spec_json, out_dir) "
                       "VALUES(?,?,?,?,?,?)", (job_id, now, now, "CREATED", json.dumps(spec), out_dir))
            db.executemany(
                "INSERT INTO lines(job_id, idx, text, lang, speaker, emotion, scene, state, updated_at) "
                "VALUES(?,?,?,?,?,?,?,'PENDING',?)",
                [(job_id, i, ln["text"], ln["lang"], ln.get("speaker"), ln.get("emotion"),
                  ln.get("scene"), now) for i, ln in enumerate(lines)])
        return True

    def job(self, job_id: str) -> sqlite3.Row | None:
        rows = self.q("SELECT * FROM jobs WHERE id=?", (job_id,))
        return rows[0] if rows else None

    def set_job_status(self, job_id: str, status: str, pause_reason: str | None = None) -> None:
        with self.tx() as db:
            db.execute("UPDATE jobs SET status=?, pause_reason=?, updated_at=? WHERE id=?",
                       (status, pause_reason, time.time(), job_id))

    def lines(self, job_id: str, states: tuple[str, ...] | None = None) -> list[sqlite3.Row]:
        if states:
            marks = ",".join("?" * len(states))
            return self.q(f"SELECT * FROM lines WHERE job_id=? AND state IN ({marks}) ORDER BY idx",
                          (job_id, *states))
        return self.q("SELECT * FROM lines WHERE job_id=? ORDER BY idx", (job_id,))

    def line(self, job_id: str, idx: int) -> sqlite3.Row:
        return self.q("SELECT * FROM lines WHERE job_id=? AND idx=?", (job_id, idx))[0]

    def set_line(self, job_id: str, idx: int, state: str, **fields) -> None:
        if state not in LINE_STATES:
            raise ValueError(f"unknown line state {state!r}")
        cols = {"state": state, "updated_at": time.time(), **fields}
        sets = ", ".join(f"{c}=?" for c in cols)
        with self.tx() as db:
            db.execute(f"UPDATE lines SET {sets} WHERE job_id=? AND idx=?", (*cols.values(), job_id, idx))

    def counts(self, job_id: str) -> dict[str, int]:
        out = dict.fromkeys(LINE_STATES, 0)
        for r in self.q("SELECT state, COUNT(*) n FROM lines WHERE job_id=? GROUP BY state", (job_id,)):
            out[r["state"]] = r["n"]
        return out

    # --- attempts & render cache --------------------------------------------
    def start_attempt(self, job_id: str, idx: int, model: str, version: str, key: str, try_no: int) -> int:
        with self.tx() as db:
            cur = db.execute("INSERT INTO attempts(job_id, idx, model, model_version, key, try_no, started_at, "
                             "outcome) VALUES(?,?,?,?,?,?,?, 'running')",
                             (job_id, idx, model, version, key, try_no, time.time()))
            return int(cur.lastrowid)

    def finish_attempt(self, attempt_id: int, outcome: str, triggers: list[str] | None = None,
                       detail: str = "", scores: dict | None = None, audio_path: str | None = None) -> None:
        with self.tx() as db:
            db.execute("UPDATE attempts SET finished_at=?, outcome=?, triggers=?, detail=?, scores=?, audio_path=? "
                       "WHERE id=?", (time.time(), outcome, json.dumps(triggers or []), detail[:2000],
                                      json.dumps(scores or {}), audio_path, attempt_id))

    def record_skip(self, job_id: str, idx: int, model: str, outcome: str, detail: str = "") -> None:
        now = time.time()
        with self.tx() as db:
            db.execute("INSERT INTO attempts(job_id, idx, model, try_no, started_at, finished_at, outcome, detail) "
                       "VALUES(?,?,?,0,?,?,?,?)", (job_id, idx, model, now, now, outcome, detail[:2000]))

    def attempts(self, job_id: str, idx: int | None = None) -> list[sqlite3.Row]:
        if idx is None:
            return self.q("SELECT * FROM attempts WHERE job_id=? ORDER BY id", (job_id,))
        return self.q("SELECT * FROM attempts WHERE job_id=? AND idx=? ORDER BY id", (job_id, idx))

    def cached_render(self, key: str) -> sqlite3.Row | None:
        rows = self.q("SELECT * FROM renders WHERE key=? AND passed=1", (key,))
        return rows[0] if rows else None

    def put_render(self, key: str, model: str, version: str, path: str, sha: str, passed: bool,
                   scores: dict) -> None:
        with self.tx() as db:
            db.execute("INSERT OR REPLACE INTO renders VALUES(?,?,?,?,?,?,?,?)",
                       (key, model, version, path, sha, int(passed), json.dumps(scores), time.time()))

    # --- self-learning --------------------------------------------------------
    def record_word(self, lang: str, word: str, model: str, ok: bool) -> None:
        with self.tx() as db:
            db.execute("INSERT INTO word_history(lang, word, model, ok, at) VALUES(?,?,?,?,?)",
                       (lang, word, model, int(ok), time.time()))

    def word_recent(self, lang: str, word: str, model: str, n: int) -> list[int]:
        rows = self.q("SELECT ok FROM word_history WHERE lang=? AND word=? AND model=? ORDER BY id DESC LIMIT ?",
                      (lang, word, model, n))
        return [r["ok"] for r in rows]

    # --- circuit breaker --------------------------------------------------------
    def record_outcome(self, model: str, lang: str, ok: bool) -> None:
        with self.tx() as db:
            db.execute("INSERT INTO model_outcomes(model, lang, ok, at) VALUES(?,?,?,?)",
                       (model, lang, int(ok), time.time()))

    def recent_outcomes(self, model: str, lang: str, n: int) -> list[int]:
        rows = self.q("SELECT ok FROM model_outcomes WHERE model=? AND lang=? ORDER BY id DESC LIMIT ?",
                      (model, lang, n))
        return [r["ok"] for r in rows]

    def clear_outcomes(self, model: str, lang: str) -> None:
        with self.tx() as db:
            db.execute("DELETE FROM model_outcomes WHERE model=? AND lang=?", (model, lang))

    def breaker(self, model: str, lang: str) -> sqlite3.Row | None:
        rows = self.q("SELECT * FROM breakers WHERE model=? AND lang=?", (model, lang))
        return rows[0] if rows else None

    def set_breaker(self, model: str, lang: str, state: str, reason: str = "",
                    opened_at: float | None = None) -> None:
        with self.tx() as db:
            db.execute("INSERT OR REPLACE INTO breakers VALUES(?,?,?,?,?)", (model, lang, state, opened_at, reason))

    # --- events -----------------------------------------------------------------
    def event(self, kind: str, job_id: str | None = None, model: str | None = None, lang: str | None = None,
              idx: int | None = None, detail: str | dict = "") -> None:
        if isinstance(detail, dict):
            detail = json.dumps(detail, ensure_ascii=False)
        with self.tx() as db:
            db.execute("INSERT INTO events(job_id, at, kind, model, lang, idx, detail) VALUES(?,?,?,?,?,?,?)",
                       (job_id, time.time(), kind, model, lang, idx, detail[:4000]))

    def events(self, job_id: str | None = None, kind: str | None = None) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM events WHERE 1=1", []
        if job_id is not None:
            sql += " AND (job_id=? OR job_id IS NULL)"
            args.append(job_id)
        if kind is not None:
            sql += " AND kind=?"
            args.append(kind)
        return self.q(sql + " ORDER BY id", args)

    # --- network task queue ---------------------------------------------------------
    def enqueue_task(self, task_id: str, kind: str, payload: dict) -> bool:
        with self.tx() as db:
            if db.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
                return False
            db.execute("INSERT INTO tasks(id, kind, payload, state, attempts, next_at, updated_at) "
                       "VALUES(?,?,?,'QUEUED',0,0,?)", (task_id, kind, json.dumps(payload), time.time()))
            return True

    def tasks(self, states: tuple[str, ...] | None = None) -> list[sqlite3.Row]:
        if states:
            marks = ",".join("?" * len(states))
            return self.q(f"SELECT * FROM tasks WHERE state IN ({marks}) ORDER BY rowid", states)
        return self.q("SELECT * FROM tasks ORDER BY rowid")

    def set_task(self, task_id: str, state: str, **fields) -> None:
        if state not in TASK_STATES:
            raise ValueError(f"unknown task state {state!r}")
        cols = {"state": state, "updated_at": time.time(), **fields}
        sets = ", ".join(f"{c}=?" for c in cols)
        with self.tx() as db:
            db.execute(f"UPDATE tasks SET {sets} WHERE id=?", (*cols.values(), task_id))
