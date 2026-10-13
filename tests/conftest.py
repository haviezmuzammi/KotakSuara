from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from app.config import REPO_ROOT, Settings
from app.engines.base import EngineInfo, ManagedModel, TTSEngine, VoiceSpec
from app.engines.model_manager import ModelManager
from app.stt.whisper import Cue

API_KEY = "test-key-0123456789abcdef-xyz"


class FakeEngine(TTSEngine):
    """Deterministic engine: 10 ms of tone per character, records every call."""

    def __init__(self, name="fake", commercial_ok=True, supports_clone=True, vram_mb=100, unofficial=False):
        self.info = EngineInfo(name=name, languages=("id",), license="test", commercial_ok=commercial_ok,
                               unofficial=unofficial, vram_mb=vram_mb, supports_clone=supports_clone)
        self.calls: list[tuple[str, VoiceSpec]] = []
        self._loaded = False
        self.device = None
        self.sr = 1000

    @property
    def loaded(self):
        return self._loaded

    def load(self, device):
        self._loaded = True
        self.device = device

    def unload(self):
        self._loaded = False

    def synthesize(self, text, voice):
        self.calls.append((text, voice))
        return np.full(len(text) * 10, 0.5, dtype=np.float32), self.sr


class FakeSTT(ManagedModel):
    name = "whisper"
    vram_mb = 50

    def __init__(self):
        self._loaded = False

    @property
    def loaded(self):
        return self._loaded

    def load(self, device):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def transcribe(self, audio_path, language="id"):
        assert Path(audio_path).exists()
        return "halo dunia", [Cue(0.0, 1.0, "halo"), Cue(1.0, 2.5, "dunia")]


@pytest.fixture
def settings(tmp_path) -> Settings:
    lexicon = tmp_path / "lexicon.yaml"
    shutil.copy(REPO_ROOT / "config" / "lexicon.yaml", lexicon)
    styles = tmp_path / "styles.yaml"
    shutil.copy(REPO_ROOT / "config" / "styles.yaml", styles)
    return Settings(api_key=API_KEY, data_dir=tmp_path / "data", lexicon_path=lexicon, styles_path=styles,
                    default_engine="fake", crossfade_ms=0, ayat_silence_ms=1000, loudness_lufs=None,
                    allowed_clients=("testclient", "127.0.0.1"))


@pytest.fixture
def fake_engine():
    return FakeEngine()


@pytest.fixture
def service(settings, fake_engine):
    from app.service import build_service

    engines = {"fake": fake_engine,
               "nc": FakeEngine("nc", commercial_ok=False, supports_clone=False, unofficial=True),
               "off": OfflineEngine()}
    return build_service(settings, engines=engines, stt=FakeSTT(),
                         manager=ModelManager(max_loaded=1, device="cuda", empty_cache=lambda: None),
                         worker_kwargs={"idle_wait": 0.02})


class OfflineEngine(FakeEngine):
    def __init__(self):
        super().__init__("off", vram_mb=0)

    def available(self):
        return False, "credentials missing"


@pytest.fixture
def client(service):
    from fastapi.testclient import TestClient

    from app.service import create_app

    with TestClient(create_app(service), headers={"X-API-Key": API_KEY}) as c:
        yield c


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False
