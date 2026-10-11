"""edge-tts (id-ID). Unofficial use of the Edge read-aloud endpoint: comparison only, may break anytime."""
from __future__ import annotations

import asyncio
import io

import numpy as np
import soundfile as sf

from app.engines.base import EngineInfo, TTSEngine, VoiceSpec

VOICES = ("id-ID-ArdiNeural", "id-ID-GadisNeural")


class EdgeIDEngine(TTSEngine):
    info = EngineInfo(
        name="edge_id",
        languages=("id",),
        license="Unofficial endpoint (no license for commercial use)",
        commercial_ok=False,
        unofficial=True,
        voices=VOICES,
        notes="Same voices as Azure, without an SLA. Do not use for published videos.",
    )

    @property
    def loaded(self) -> bool:
        return True

    def load(self, device: str) -> None:
        pass

    def unload(self) -> None:
        pass

    def synthesize(self, text: str, voice: VoiceSpec) -> tuple[np.ndarray, int]:
        import edge_tts

        name = voice.voice or VOICES[0]
        if name not in VOICES:
            raise ValueError(f"unknown edge voice {name!r}; choose one of {VOICES}")

        async def fetch() -> bytes:
            buffer = bytearray()
            async for item in edge_tts.Communicate(text, name).stream():
                if item["type"] == "audio":
                    buffer.extend(item["data"])
            return bytes(buffer)

        audio, sr = sf.read(io.BytesIO(asyncio.run(fetch())), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
