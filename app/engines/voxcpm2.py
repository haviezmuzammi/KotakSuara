"""VoxCPM2 (OpenBMB, Apache-2.0): native Indonesian, 48 kHz, voice cloning and natural-language style instructions.

Its dependencies (transformers, gradio, funasr, ...) clash with Chatterbox's pins, so it lives in a separate Python
environment (VOXCPM_PYTHON) and runs as a child process (voxcpm_worker.py) that talks JSON lines over pipes.
Loading starts the process, unloading ends it, which frees its VRAM completely for the next engine.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import threading
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from app.engines.base import EngineInfo, TTSEngine, VoiceSpec

log = logging.getLogger(__name__)

WORKER = Path(__file__).with_name("voxcpm_worker.py")
_PARAMS = {"cfg_value": float, "inference_timesteps": int, "seed": int}
_BRACKETS = re.compile(r"\s*[()]\s*")
_COMMA_BEFORE_PUNCT = re.compile(r",\s*([.,!?;:])")


def engine_text(text: str, instruction: str | None) -> str:
    """VoxCPM2 reads a leading "(...)" as a delivery instruction, so brackets in the script itself become commas
    (otherwise "(HR. Bukhari) ..." at the start of a chunk would be swallowed as an instruction)."""
    text = _COMMA_BEFORE_PUNCT.sub(r"\1", _BRACKETS.sub(", ", text)).strip(" ,") or text
    return f"({instruction}){text}" if instruction else text


class VoxCPM2Engine(TTSEngine):
    info = EngineInfo(
        name="voxcpm2",
        languages=("id",),
        license="Apache-2.0",
        commercial_ok=True,
        vram_mb=6500,
        supports_clone=True,
        notes="Native Indonesian. Reading styles become spoken-style instructions. About 2x slower than Chatterbox.",
    )

    def __init__(self, python: str, hf_home: str = "", tmp_dir: Path | None = None,
                 load_timeout: float = 600, call_timeout: float = 900, worker: Path = WORKER):
        self.python = python
        self.worker = Path(worker)
        self.hf_home = hf_home
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None
        self.load_timeout = load_timeout
        self.call_timeout = call_timeout
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._device = "cuda"
        self.sample_rate = 48000

    def available(self) -> tuple[bool, str]:
        if not self.python:
            return False, "VOXCPM_PYTHON not set"
        if not Path(self.python).exists():
            return False, f"VOXCPM_PYTHON not found: {self.python}"
        return True, ""

    @property
    def loaded(self) -> bool:
        return self._proc is not None

    def load(self, device: str) -> None:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(reason)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"}
        if self.hf_home:
            env["HF_HOME"] = self.hf_home
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._lines = queue.Queue()
        self._proc = subprocess.Popen([self.python, str(self.worker), device], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                      text=True, encoding="utf-8", creationflags=flags)
        self._device = device
        threading.Thread(target=self._read, args=(self._proc.stdout, self._lines), daemon=True).start()
        threading.Thread(target=self._drain, args=(self._proc.stderr,), daemon=True).start()
        first = self._next(self.load_timeout)
        if not first.get("ready"):
            self.unload()
            raise RuntimeError(f"VoxCPM2 failed to load: {first.get('error', first)}")
        self.sample_rate = int(first["sample_rate"])

    @staticmethod
    def _read(stream, lines: queue.Queue) -> None:
        for line in stream:
            lines.put(line)
        lines.put(None)  # EOF: the worker exited

    @staticmethod
    def _drain(stream) -> None:
        for line in stream:  # library logs and tracebacks go to the service log
            if line.strip() and "it/s]" not in line and "s/it]" not in line:
                log.info("voxcpm: %s", line.rstrip()[:300])

    def _next(self, timeout: float) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            self.unload()
            raise RuntimeError(f"VoxCPM2 did not answer within {timeout:.0f} s") from None
        if line is None:
            self.unload()
            raise RuntimeError("VoxCPM2 worker exited unexpectedly")
        return json.loads(line)

    def unload(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait(timeout=10)

    def synthesize(self, text: str, voice: VoiceSpec) -> tuple[np.ndarray, int]:
        if self._proc is None or self._proc.poll() is not None:  # crashed since the last call: start again
            self._proc = None
            self.load(self._device)
        tmp_dir = self.tmp_dir or Path(os.environ.get("TEMP", "/tmp"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out = tmp_dir / f"voxcpm-{uuid.uuid4().hex}.wav"
        request = {"text": engine_text(text, voice.instruction), "out": str(out)}
        request.update({k: cast(voice.params[k]) for k, cast in _PARAMS.items() if k in voice.params})
        if voice.reference_path:
            request["reference"] = str(voice.reference_path)
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        reply = self._next(self.call_timeout)
        try:
            if not reply.get("ok"):
                raise RuntimeError(f"VoxCPM2: {reply.get('error')}")  # "out of memory" -> CPU retry upstream
            audio, sr = sf.read(out, dtype="float32")
        finally:
            out.unlink(missing_ok=True)
        return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)

