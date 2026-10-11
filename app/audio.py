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

    @property
    def ends_with_speech(self) -> bool:
        return self._last_was_speech

    def add_gap(self, ms: int) -> None:
        """Short pause between parts of one sentence (no marker, keeps the next speech from crossfading)."""
        samples = int(self.sample_rate * ms / 1000)
        self._parts.append(np.zeros(samples, dtype=np.float32))
        self._length += samples
        self._last_was_speech = False

    def add_speech(self, audio: np.ndarray, sample_rate: int, crossfade: bool = True) -> None:
        if sample_rate != self.sample_rate:
            raise ValueError(f"sample rate mismatch: {sample_rate} != {self.sample_rate}")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        fade = int(self.sample_rate * self.crossfade_ms / 1000)
        if crossfade and self._last_was_speech and fade > 0 and self._parts:
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


def trim_silence(audio: np.ndarray, sample_rate: int, threshold_db: float = -42.0, keep_ms: int = 30) -> np.ndarray:
    """Cut leading/trailing near-silence (engines pad each call), keeping a few ms so onsets stay natural."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    window = max(1, int(sample_rate * 0.01))
    frames = audio[: audio.size // window * window].reshape(-1, window)
    if frames.size == 0:
        return audio
    level = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    loud = np.flatnonzero(level > threshold_db)
    if loud.size == 0:
        return audio
    keep = int(sample_rate * keep_ms / 1000)
    start = max(0, loud[0] * window - keep)
    end = min(audio.size, (loud[-1] + 1) * window + keep)
    return audio[start:end]
