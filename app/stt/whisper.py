"""faster-whisper speech-to-text with SRT output."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.engines.base import ManagedModel


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def _timestamp(seconds: float) -> str:
    millis = max(0, int(round(seconds * 1000)))
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks = [f"{i}\n{_timestamp(c.start)} --> {_timestamp(c.end)}\n{c.text.strip()}\n"
              for i, c in enumerate(cues, start=1)]
    return "\n".join(blocks)


class WhisperSTT(ManagedModel):
    name = "whisper"
    vram_mb = 2000

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, device: str) -> None:
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(self.model_id, device=device, compute_type=compute_type)

    def unload(self) -> None:
        self._model = None

    def transcribe(self, audio_path: Path, language: str = "id") -> tuple[str, list[Cue]]:
        segments, _info = self._model.transcribe(str(audio_path), language=language, vad_filter=True)
        cues = [Cue(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
        return " ".join(c.text for c in cues), cues
