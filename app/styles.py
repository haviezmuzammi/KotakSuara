"""Reading styles per genre (config/styles.yaml): engine params, a spoken-style instruction for engines that take
one (VoxCPM2), pauses between chunks, optional audio effect. Also final loudness normalization (ffmpeg loudnorm).

Hot-reloaded like the lexicon: the file is re-read when its modification time changes and an invalid file is
rejected while the last good version stays in use.
"""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger(__name__)

_NAME = re.compile(r"^[a-z][a-z0-9_-]{1,30}$")
_PARAM_NAMES = {"exaggeration", "cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p"}


@dataclass(frozen=True)
class Style:
    name: str
    label: str
    description: str = ""
    params: dict[str, float] = field(default_factory=dict)
    gap_ms: int | None = None
    fx: str | None = None
    instruction: str | None = None  # natural-language delivery description for instruction-following engines

    def as_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "description": self.description,
                "params": dict(self.params), "gap_ms": self.gap_ms, "fx": self.fx, "instruction": self.instruction}


def parse_styles(text: str) -> dict[str, Style]:
    raw = yaml.safe_load(text) or {}
    entries = raw.get("styles") if isinstance(raw, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ValueError("styles.yaml needs a non-empty 'styles' mapping")
    out: dict[str, Style] = {}
    for name, spec in entries.items():
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError(f"invalid style name {name!r} (lowercase letters, digits, - and _)")
        spec = spec or {}
        if not isinstance(spec, dict):
            raise ValueError(f"style {name!r} must be a mapping")
        params = spec.get("params") or {}
        if not isinstance(params, dict) or not set(params) <= _PARAM_NAMES:
            raise ValueError(f"style {name!r}: params must use {sorted(_PARAM_NAMES)}")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in params.values()):
            raise ValueError(f"style {name!r}: params must be numbers")
        gap = spec.get("gap_ms")
        if gap is not None and (not isinstance(gap, int) or not 0 <= gap <= 5000):
            raise ValueError(f"style {name!r}: gap_ms must be an integer 0-5000")
        fx = spec.get("fx")
        if fx is not None and (not isinstance(fx, str) or not fx.strip()):
            raise ValueError(f"style {name!r}: fx must be a non-empty ffmpeg filter string")
        instruction = spec.get("instruction")
        if instruction is not None and (not isinstance(instruction, str) or not 0 < len(instruction.strip()) <= 300
                                        or set(instruction) & set("()")):
            raise ValueError(f"style {name!r}: instruction must be 1-300 characters without parentheses")
        out[name] = Style(name=name, label=str(spec.get("label") or name.title()),
                          description=str(spec.get("description") or ""),
                          params={k: float(v) for k, v in params.items()}, gap_ms=gap,
                          fx=fx.strip() if fx else None, instruction=instruction.strip() if instruction else None)
    return out


class Styles:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._styles: dict[str, Style] = {}
        self.last_error: str | None = None
        self.all()
        if self.last_error is not None and not self._styles:
            raise FileNotFoundError(f"styles not found or invalid: {self.path} ({self.last_error})")

    def all(self) -> dict[str, Style]:
        with self._lock:
            try:
                mtime = self.path.stat().st_mtime
            except OSError:
                self.last_error = f"cannot read {self.path}"
                return self._styles
            if mtime != self._mtime:
                try:
                    self._styles = parse_styles(self.path.read_text(encoding="utf-8"))
                    self.last_error = None
                    log.info("styles loaded: %s", ", ".join(self._styles))
                except (yaml.YAMLError, ValueError) as exc:
                    self.last_error = str(exc)
                    log.error("styles rejected, keeping previous version: %s", exc)
                self._mtime = mtime
            return self._styles

    def get(self, name: str) -> Style | None:
        return self.all().get(name.strip().lower())


class EffectError(RuntimeError):
    pass


def find_ffmpeg(configured: str = "") -> str | None:
    if configured:
        return configured if Path(configured).exists() else None
    return shutil.which("ffmpeg")


def _ffmpeg(audio: np.ndarray, sample_rate: int, chain: str, ffmpeg: str, output: list[str], timeout: int,
            loglevel: str = "error") -> subprocess.CompletedProcess:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", loglevel, "-nostdin",
           "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0", "-af", chain, *output]
    try:
        proc = subprocess.run(cmd, input=np.asarray(audio, dtype="<f4").tobytes(), capture_output=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EffectError(f"ffmpeg could not run: {exc}") from None
    if proc.returncode != 0:
        raise EffectError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()[-300:]}")
    return proc


def apply_fx(audio: np.ndarray, sample_rate: int, chain: str, ffmpeg: str, timeout: int = 300) -> np.ndarray:
    """Run mono float32 audio through an ffmpeg filter chain (raw PCM over pipes, no temp files, no shell)."""
    proc = _ffmpeg(audio, sample_rate, chain, ffmpeg,
                   ["-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "pipe:1"], timeout)
    out = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if out.size == 0:
        raise EffectError("ffmpeg returned no audio")
    peak = float(np.max(np.abs(out)))
    return out / peak if peak > 1.0 else out


def normalize_loudness(audio: np.ndarray, sample_rate: int, ffmpeg: str, target_lufs: float = -16.0,
                       true_peak_db: float = -1.5, timeout: int = 300) -> tuple[np.ndarray, float | None]:
    """Two-pass EBU R128 normalization (ffmpeg loudnorm): measure, then correct to `target_lufs`.

    Engines differ a lot (VoxCPM2 "sad" delivery came out near -35 LUFS, Chatterbox near -17 with peaks at 0 dBFS),
    so every narration is brought to the same level. Returns the audio and the measured input loudness;
    silent audio is returned unchanged.
    """
    measure = _ffmpeg(audio, sample_rate, f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA=11:print_format=json",
                      ffmpeg, ["-f", "null", "-"], timeout, loglevel="info")
    text = measure.stderr.decode(errors="replace")
    try:
        stats = json.loads(text[text.rindex("{"):text.rindex("}") + 1])
        measured = {k: float(stats[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    except (ValueError, KeyError) as exc:
        raise EffectError(f"could not read loudness measurement: {exc}") from None
    if not all(math.isfinite(v) for v in measured.values()) or measured["input_i"] < -70:
        return audio, None  # silence: nothing to normalize
    chain = (f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA=11:measured_I={measured['input_i']}"
             f":measured_TP={measured['input_tp']}:measured_LRA={measured['input_lra']}"
             f":measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:linear=true")
    out = apply_fx(audio, sample_rate, chain, ffmpeg, timeout)
    return out, measured["input_i"]
