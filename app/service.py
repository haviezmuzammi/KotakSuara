"""Wires settings, storage, engines, the GPU worker and the API together."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI

from app.api import build_api
from app.config import Settings
from app.db import Database
from app.engines.base import ManagedModel, TTSEngine
from app.engines.model_manager import ModelManager
from app.jobs import JobProcessor
from app.power import PowerGuard
from app.text.lexicon import Lexicon
from app.worker import JobWorker


@dataclass
class Service:
    settings: Settings
    db: Database
    engines: dict[str, TTSEngine]
    manager: ModelManager
    lexicon: Lexicon
    worker: JobWorker


def setup_logging(settings: Settings) -> None:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(settings.logs_dir / "service.log", maxBytes=5_000_000, backupCount=5,
                                  encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()],
                        format="%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s")


def build_service(settings: Settings, engines: dict[str, TTSEngine] | None = None,
                  stt: ManagedModel | None = None, manager: ModelManager | None = None,
                  power: PowerGuard | None = None, worker_kwargs: dict | None = None) -> Service:
    if engines is None:
        from app.engines.registry import build_engines

        engines = build_engines(settings)
    if stt is None:
        from app.stt.whisper import WhisperSTT

        stt = WhisperSTT(settings.stt_model)
    for path in (settings.jobs_dir, settings.profiles_dir, settings.logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    db = Database(settings.db_path)
    manager = manager or ModelManager(max_loaded=settings.max_loaded_models)
    lexicon = Lexicon(settings.lexicon_path)
    processor = JobProcessor(settings, db, engines, manager, stt, lexicon)
    worker = JobWorker(db, processor, power or PowerGuard(), stuck_seconds=settings.stuck_job_seconds,
                       **(worker_kwargs or {}))
    return Service(settings, db, engines, manager, lexicon, worker)


def create_app(svc: Service) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        svc.worker.start()
        try:
            yield
        finally:
            svc.worker.stop()

    app = build_api(svc)
    app.router.lifespan_context = lifespan
    return app
