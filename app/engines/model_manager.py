"""Keeps at most N heavy models on the GPU (LRU) and falls back to CPU on out-of-memory."""
from __future__ import annotations

import gc
import logging
import threading
from collections import OrderedDict
from typing import Callable, TypeVar

from app.engines.base import ManagedModel

log = logging.getLogger(__name__)
T = TypeVar("T")


def is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return "out of memory" in str(exc).lower()


def default_empty_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


class ModelManager:
    def __init__(self, max_loaded: int = 1, device: str | None = None,
                 empty_cache: Callable[[], None] = default_empty_cache):
        self.max_loaded = max(1, max_loaded)
        self.preferred_device = device or default_device()
        self._empty_cache = empty_cache
        self._loaded: OrderedDict[str, tuple[ManagedModel, str]] = OrderedDict()
        self._lock = threading.RLock()
        self._snapshot: list[dict] = []

    @property
    def loaded(self) -> list[dict]:
        """Lock-free view, so /health never waits for a running synthesis."""
        return list(self._snapshot)

    def _publish(self) -> None:
        self._snapshot = [{"name": name, "device": device} for name, (_, device) in self._loaded.items()]

    def _unload(self, name: str) -> None:
        model, device = self._loaded.pop(name)
        log.info("unloading %s from %s", name, device)
        self._publish()
        model.unload()
        self._empty_cache()

    def unload_all(self) -> None:
        with self._lock:
            for name in list(self._loaded):
                self._unload(name)

    def release_fallbacks(self) -> None:
        """Drop models that fell back to CPU so the next job tries the GPU again."""
        with self._lock:
            for name, (_, device) in list(self._loaded.items()):
                if device != self.preferred_device:
                    self._unload(name)

    def ensure(self, model: ManagedModel, device: str | None = None) -> str:
        """Load `model` (evicting least-recently-used heavy models) and return its device."""
        with self._lock:
            target = device or self.preferred_device
            if not model.heavy:
                if not model.loaded:
                    model.load("cpu")
                return "cpu"
            if model.name in self._loaded:
                current = self._loaded[model.name][1]
                if device is None or current == target:
                    self._loaded.move_to_end(model.name)
                    self._publish()
                    return current
                self._unload(model.name)
            while len(self._loaded) >= self.max_loaded:
                self._unload(next(iter(self._loaded)))
            log.info("loading %s on %s", model.name, target)
            model.load(target)
            self._loaded[model.name] = (model, target)
            self._publish()
            return target

    def run(self, model: ManagedModel, fn: Callable[[ManagedModel], T]) -> T:
        """Run `fn(model)` with the model loaded; on GPU OOM retry once on CPU."""
        with self._lock:
            try:
                self.ensure(model)
                return fn(model)
            except Exception as exc:
                if not (model.heavy and is_oom(exc)):
                    raise
                log.warning("OOM on %s, retrying on CPU: %s", model.name, exc)
                self.unload_all()
                self.ensure(model, device="cpu")
                return fn(model)
