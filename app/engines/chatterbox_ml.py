"""Chatterbox Multilingual (Resemble AI, MIT). Indonesian text is spoken with the Malay (`ms`) model.

Output carries Resemble's imperceptible Perth watermark (applied by the library).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.engines.base import EngineInfo, TTSEngine, VoiceSpec

_PARAMS = {"exaggeration", "cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p"}


class ChatterboxMLEngine(TTSEngine):
    info = EngineInfo(
        name="chatterbox_ml",
        languages=("id (via ms)",),
        license="MIT",
        commercial_ok=True,
        vram_mb=3500,
        supports_clone=True,
        extra_languages=("ar",),
        notes="Zero-shot clone from a consented reference clip (5-15 s, clean speech).",
    )
    language_id = "ms"

    def __init__(self) -> None:
        self._model = None
        self._builtin_conds = None
        self._conds_key: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, device: str) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        self._builtin_conds = self._model.conds
        self._conds_key = None

    def unload(self) -> None:
        self._model = None
        self._builtin_conds = None
        self._conds_key = None

    def _select_voice(self, reference: Path | None, exaggeration: float) -> None:
        key = str(reference) if reference else None
        if key == self._conds_key:
            return
        if reference:
            self._model.prepare_conditionals(str(reference), exaggeration=exaggeration)
        else:
            self._model.conds = self._builtin_conds
        self._conds_key = key

    def synthesize(self, text: str, voice: VoiceSpec) -> tuple[np.ndarray, int]:
        import torch

        params = {k: float(v) for k, v in voice.params.items() if k in _PARAMS}
        if "seed" in voice.params:
            torch.manual_seed(int(voice.params["seed"]))
        self._select_voice(voice.reference_path, params.get("exaggeration", 0.5))
        with torch.inference_mode():
            wav = self._model.generate(text, language_id=voice.language or self.language_id, **params)
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32), int(self._model.sr)
