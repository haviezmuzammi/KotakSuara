"""Engine registry: add a new engine here after creating its module."""
from __future__ import annotations

from app.config import Settings
from app.engines.azure_id import AzureIDEngine
from app.engines.base import TTSEngine
from app.engines.chatterbox_ml import ChatterboxMLEngine
from app.engines.edge_id import EdgeIDEngine
from app.engines.mms_id import MMSIDEngine


def build_engines(settings: Settings) -> dict[str, TTSEngine]:
    engines: list[TTSEngine] = [
        ChatterboxMLEngine(),
        AzureIDEngine(settings.azure_key, settings.azure_region),
        MMSIDEngine(),
    ]
    if settings.enable_unofficial_engines:
        engines.append(EdgeIDEngine())
    return {e.info.name: e for e in engines}
