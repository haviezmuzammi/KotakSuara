"""Keep Windows awake only while jobs run.

SetThreadExecutionState is per-thread: the flag must be set and cleared from the same thread,
which is why only the GPU worker thread calls these methods.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Callable

log = logging.getLogger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _default_setter() -> Callable[[int], int]:
    if sys.platform == "win32":
        import ctypes

        fn = ctypes.windll.kernel32.SetThreadExecutionState
        fn.argtypes = [ctypes.c_uint32]
        fn.restype = ctypes.c_uint32
        return fn
    return lambda flags: 1


class PowerGuard:
    def __init__(self, setter: Callable[[int], int] | None = None):
        self._set = setter or _default_setter()
        self.owner_thread: int | None = None
        self.active = False

    def keep_awake(self) -> None:
        if self.active:
            return
        if self._set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) == 0:
            log.error("SetThreadExecutionState(keep awake) failed")
        self.owner_thread = threading.get_ident()
        self.active = True

    def release(self) -> None:
        if not self.active:
            return
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("PowerGuard.release() must run on the thread that called keep_awake()")
        self._set(ES_CONTINUOUS)
        self.active = False
