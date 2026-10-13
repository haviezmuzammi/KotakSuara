import io
import urllib.error

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.engines.azure_id import AzureIDEngine, build_ssml
from app.engines.base import VoiceSpec
from app.engines.registry import build_engines


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(2400, dtype=np.float32), 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_build_ssml_escapes_text():
    ssml = build_ssml("A & B <c>", "id-ID-GadisNeural", rate="-5%")
    assert "A &amp; B &lt;c&gt;" in ssml and "name=\"id-ID-GadisNeural\"" in ssml and "rate=\"-5%\"" in ssml


def test_azure_retries_then_succeeds():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "busy", {}, None)
        return _Response(_wav_bytes())

    engine = AzureIDEngine("k", "southeastasia", opener=opener, sleep=lambda s: None)
    audio, sr = engine.synthesize("Halo", VoiceSpec())
    assert sr == 24000 and audio.size == 2400 and len(calls) == 2
    assert calls[0].full_url.startswith("https://southeastasia.tts.speech.microsoft.com/")
    assert engine.characters_billed == 4


def test_azure_errors():
    def denied(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "denied", {}, None)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        AzureIDEngine("k", "r", opener=denied).synthesize("x", VoiceSpec())
    with pytest.raises(ValueError):
        AzureIDEngine("k", "r").synthesize("x", VoiceSpec(voice="en-US-Foo"))
    engine = AzureIDEngine("", "")
    assert engine.available()[0] is False
    with pytest.raises(RuntimeError):
        engine.synthesize("x", VoiceSpec())


def test_registry_flags():
    engines = build_engines(Settings(api_key="x" * 24))
    assert set(engines) == {"chatterbox_ml", "azure_id", "mms_id", "edge_id", "voxcpm2"}
    assert engines["mms_id"].info.commercial_ok is False
    assert engines["edge_id"].info.unofficial is True
    assert engines["chatterbox_ml"].info.supports_clone is True
    assert "edge_id" not in build_engines(Settings(api_key="x" * 24, enable_unofficial_engines=False))
