"""SQLite storage for jobs and voice profiles."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    request TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    result TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    heartbeat_at TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    engine TEXT NOT NULL,
    reference_path TEXT NOT NULL,
    consent_by TEXT NOT NULL,
    consent_date TEXT NOT NULL,
    consent_evidence TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL
);
"""

MAX_ATTEMPTS = 2


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _job(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    job = dict(row)
    job["request"] = json.loads(job["request"])
    job["result"] = json.loads(job["result"]) if job["result"] else None
    return job


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # jobs -----------------------------------------------------------------
    def create_job(self, kind: str, request: dict) -> str:
        job_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute("INSERT INTO jobs (id, kind, status, request, created_at) VALUES (?, ?, 'queued', ?, ?)",
                         (job_id, kind, json.dumps(request), now()))
        return job_id

    def get_job(self, job_id: str) -> dict | None:
        with self._connect() as conn:
            return _job(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def claim_next(self) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, rowid LIMIT 1").fetchone()
            if row is None:
                return None
            ts = now()
            conn.execute("UPDATE jobs SET status = 'running', attempts = attempts + 1, started_at = ?, "
                         "heartbeat_at = ?, error = NULL WHERE id = ?", (ts, ts, row["id"]))
            return _job(conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone())

    def heartbeat(self, job_id: str, progress: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE jobs SET progress = ?, heartbeat_at = ? WHERE id = ?",
                         (round(progress, 4), now(), job_id))

    def finish_job(self, job_id: str, result: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE jobs SET status = 'done', progress = 1, result = ?, finished_at = ? WHERE id = ?",
                         (json.dumps(result), now(), job_id))

    def fail_job(self, job_id: str, error: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE jobs SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
                         (error[:2000], now(), job_id))

    def recover_interrupted(self) -> dict[str, int]:
        """Jobs left 'running' by a crash/restart are re-queued once, then failed."""
        with self._lock, self._connect() as conn:
            failed = conn.execute(
                "UPDATE jobs SET status = 'failed', error = 'interrupted too many times', finished_at = ? "
                "WHERE status = 'running' AND attempts >= ?", (now(), MAX_ATTEMPTS)).rowcount
            requeued = conn.execute(
                "UPDATE jobs SET status = 'queued', progress = 0 WHERE status = 'running'").rowcount
        return {"requeued": requeued, "failed": failed}

    def queue_length(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()[0]

    # profiles -------------------------------------------------------------
    def create_profile(self, profile: dict) -> dict:
        record = {"id": uuid.uuid4().hex, "created_at": now(), **profile}
        cols = ", ".join(record)
        marks = ", ".join("?" for _ in record)
        with self._lock, self._connect() as conn:
            conn.execute(f"INSERT INTO profiles ({cols}) VALUES ({marks})", tuple(record.values()))
        return record

    def get_profile(self, profile_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        return dict(row) if row else None

    def list_profiles(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM profiles ORDER BY created_at")]

    def delete_profile(self, profile_id: str) -> bool:
        with self._lock, self._connect() as conn:
            return conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,)).rowcount > 0
