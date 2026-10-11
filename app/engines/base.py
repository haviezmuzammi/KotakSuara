"""Engine contracts. A new TTS engine or voice converter is a new file implementing one of these."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EngineInfo:
    name: str
    languages: tuple[str, ...]
    license: str
    commercial_ok: bool
    unofficial: bool = False
    vram_mb: int = 0  # 0 = no GPU model (cloud API or CPU-only)
    supports_clone: bool = False
    voices: tuple[str, ...] = ()
    notes: str = ""

    def as_dict(self) -> dict:
        data = asdict(self)
        data["languages"] = list(self.languages)
        data["voices"] = list(self.voices)
        return data


@dataclass(frozen=True)
class VoiceSpec:
    voice: str | None = None            # named voice for cloud engines
    reference_path: Path | None = None  # consented reference audio for cloning
    params: dict = field(default_factory=dict)


class ManagedModel(ABC):
    """Anything the ModelManager can load onto / unload from a device."""

    name: str = ""
    vram_mb: int = 0

    @property
    def heavy(self) -> bool:
        return self.vram_mb > 0

    @abstractmethod
    def load(self, device: str) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @property
    @abstractmethod
    def loaded(self) -> bool: ...


class TTSEngine(ManagedModel):
    info: EngineInfo

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.info.name

    @property
    def vram_mb(self) -> int:  # type: ignore[override]
        return self.info.vram_mb

    def available(self) -> tuple[bool, str]:
        """Whether the engine can run now (e.g. credentials present)."""
        return True, ""

    @abstractmethod
    def synthesize(self, text: str, voice: VoiceSpec) -> tuple[np.ndarray, int]:
        """Return mono float32 audio in [-1, 1] and its sample rate."""


class VoiceConverter(ManagedModel):
    """Changes the timbre of already-synthesized audio to match a reference voice.

    No converter ships today (OpenVoice v2 was dropped: its pins conflict with Chatterbox and its
    av==10 dependency does not build on Python 3.11 for Windows). Implement this to add one.
    """

    license: str = ""
    commercial_ok: bool = False

    @abstractmethod
    def convert(self, audio: np.ndarray, sample_rate: int, reference_path: Path) -> tuple[np.ndarray, int]: ...
