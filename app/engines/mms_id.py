"""facebook/mms-tts-ind (VITS). CC-BY-NC 4.0: comparison only, never for monetized videos."""
from __future__ import annotations

import numpy as np

from app.engines.base import EngineInfo, TTSEngine, VoiceSpec

MODEL_ID = "facebook/mms-tts-ind"


class MMSIDEngine(TTSEngine):
    info = EngineInfo(
        name="mms_id",
        languages=("id",),
        license="CC-BY-NC-4.0",
        commercial_ok=False,
        vram_mb=300,
        notes="Non-commercial license. Single built-in voice, no cloning.",
    )

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = "cpu"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, device: str) -> None:
        from transformers import AutoTokenizer, VitsModel

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = VitsModel.from_pretrained(MODEL_ID).to(device).eval()
        self._device = device

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None

    def synthesize(self, text: str, voice: VoiceSpec) -> tuple[np.ndarray, int]:
        import torch

        if "seed" in voice.params:
            torch.manual_seed(int(voice.params["seed"]))
        inputs = self._tokenizer(text.lower(), return_tensors="pt").to(self._device)
        with torch.inference_mode():
            waveform = self._model(**inputs).waveform
        return waveform.squeeze(0).float().cpu().numpy(), int(self._model.config.sampling_rate)
