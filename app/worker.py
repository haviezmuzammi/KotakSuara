"""Serial job queue served by ONE long-lived worker thread (all GPU work happens here)."""
from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Callable

from app.db import Database
from app.power import PowerGuard

log = logging.getLogger(__name__)

Processor = Callable[[dict, Callable[[float], None]], dict]


class JobError(Exception):
    """An expected job failure whose message is safe to show to API clients."""


class JobWorker:
    def __init__(self, db: Database, processor: Processor, power: PowerGuard,
                 stuck_seconds: int = 900, on_stuck: Callable[[], None] | None = None,
                 idle_wait: float = 1.0):
        self.db = db
        self.processor = processor
        self.power = power
        self.stuck_seconds = stuck_seconds
        self.on_stuck = on_stuck or (lambda: os._exit(3))  # NSSM restarts the service
        self.idle_wait = idle_wait
        self.current_job: str | None = None
        self.thread_id: int | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        self._last_beat = time.monotonic()

    @property
    def busy(self) -> bool:
        return self.current_job is not None or self.db.queue_length() > 0

    def start(self) -> None:
        recovered = self.db.recover_interrupted()
        if any(recovered.values()):
            log.warning("recovered interrupted jobs: %s", recovered)
        self._thread = threading.Thread(target=self._run, name="gpu-worker", daemon=True)
        self._thread.start()
        self._watchdog = threading.Thread(target=self._watch, name="watchdog", daemon=True)
        self._watchdog.start()

    def stop(self, timeout: float = 10) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def notify(self) -> None:
        self._wake.set()

    def _beat(self, job_id: str, progress: float) -> None:
        self._last_beat = time.monotonic()
        self.db.heartbeat(job_id, progress)

    def _run(self) -> None:
        self.thread_id = threading.get_ident()
        while not self._stop.is_set():
            job = self.db.claim_next()
            if job is None:
                self.power.release()
                self._wake.wait(self.idle_wait)
                self._wake.clear()
                continue
            self.power.keep_awake()
            self.current_job = job["id"]
            self._last_beat = time.monotonic()
            started = time.monotonic()
            try:
                result = self.processor(job, lambda p, jid=job["id"]: self._beat(jid, p))
                result["elapsed_s"] = round(time.monotonic() - started, 2)
                self.db.finish_job(job["id"], result)
                log.info("job %s done in %.1fs", job["id"], result["elapsed_s"])
            except JobError as exc:
                self.db.fail_job(job["id"], str(exc))
                log.warning("job %s failed: %s", job["id"], exc)
            except Exception as exc:  # noqa: BLE001 - keep the worker alive
                self.db.fail_job(job["id"], f"{type(exc).__name__}: {exc}")
                log.error("job %s crashed:\n%s", job["id"], traceback.format_exc())
            finally:
                self.current_job = None
        self.power.release()

    def _watch(self) -> None:
        while not self._stop.wait(min(30.0, self.stuck_seconds / 3)):
            if self.current_job and time.monotonic() - self._last_beat > self.stuck_seconds:
                log.critical("job %s made no progress for %ss; restarting process (%s)",
                             self.current_job, self.stuck_seconds, datetime.now(timezone.utc).isoformat())
                self.on_stuck()
                return
