"""Audio assembly: crossfaded speech chunks, silence for ayat blocks, timestamps."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Marker:
    index: int
    start_s: float
    end_s: float
    text: str
    source: str


@dataclass
class AudioBuilder:
    sample_rate: int
    crossfade_ms: int = 50
    _parts: list[np.ndarray] = field(default_factory=list)
    _length: int = 0
    _last_was_speech: bool = False
    markers: list[Marker] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self._length / self.sample_rate

    def add_speech(self, audio: np.ndarray, sample_rate: int) -> None:
        if sample_rate != self.sample_rate:
            raise ValueError(f"sample rate mismatch: {sample_rate} != {self.sample_rate}")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        fade = int(self.sample_rate * self.crossfade_ms / 1000)
        if self._last_was_speech and fade > 0 and self._parts:
            prev = self._parts[-1]
            fade = min(fade, prev.size, audio.size)
            if fade > 0:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                mixed = prev[-fade:] * (1.0 - ramp) + audio[:fade] * ramp
                self._parts[-1] = np.concatenate([prev[:-fade], mixed])
                audio = audio[fade:]  # the overlapped head now lives inside the previous part
        self._parts.append(audio)
        self._length += audio.size
        self._last_was_speech = True

    def add_silence(self, ms: int, marker_text: str | None = None, source: str = "") -> None:
        samples = int(self.sample_rate * ms / 1000)
        start = self.duration_s
        self._parts.append(np.zeros(samples, dtype=np.float32))
        self._length += samples
        self._last_was_speech = False
        if marker_text is not None:
            self.markers.append(Marker(len(self.markers) + 1, round(start, 3), round(self.duration_s, 3),
                                       marker_text, source))

    def build(self) -> np.ndarray:
        if not self._parts:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._parts)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return audio.astype(np.float32)
