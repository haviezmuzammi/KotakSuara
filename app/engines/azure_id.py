"""Azure Neural TTS (id-ID) via REST + SSML. Official, commercial use allowed; billed per character."""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import soundfile as sf

from app.engines.base import EngineInfo, TTSEngine, VoiceSpec

VOICES = ("id-ID-ArdiNeural", "id-ID-GadisNeural")


def build_ssml(text: str, voice: str, rate: str | None = None, pitch: str | None = None) -> str:
    body = escape(text)
    prosody = {k: v for k, v in (("rate", rate), ("pitch", pitch)) if v}
    if prosody:
        attrs = " ".join(f"{k}={quoteattr(v)}" for k, v in prosody.items())
        body = f"<prosody {attrs}>{body}</prosody>"
    return (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='id-ID'>"
            f"<voice name={quoteattr(voice)}>{body}</voice></speak>")


class AzureIDEngine(TTSEngine):
    info = EngineInfo(
        name="azure_id",
        languages=("id",),
        license="Azure Speech service terms (paid API)",
        commercial_ok=True,
        voices=VOICES,
        notes="Quality baseline. Needs AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env.",
    )

    def __init__(self, key: str, region: str, opener=urllib.request.urlopen, sleep=time.sleep):
        self._key = key
        self._region = region
        self._open = opener
        self._sleep = sleep
        self.characters_billed = 0

    @property
    def loaded(self) -> bool:
        return True

    def load(self, device: str) -> None:
        pass

    def unload(self) -> None:
        pass

    def available(self) -> tuple[bool, str]:
        if not (self._key and self._region):
            return False, "AZURE_SPEECH_KEY / AZURE_SPEECH_REGION not set"
        return True, ""

    def synthesize(self, text: str, voice: VoiceSpec) -> tuple[np.ndarray, int]:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(reason)
        name = voice.voice or VOICES[0]
        if name not in VOICES:
            raise ValueError(f"unknown Azure voice {name!r}; choose one of {VOICES}")
        ssml = build_ssml(text, name, voice.params.get("rate"), voice.params.get("pitch"))
        request = urllib.request.Request(
            f"https://{self._region}.tts.speech.microsoft.com/cognitiveservices/v1",
            data=ssml.encode("utf-8"),
            method="POST",
            headers={
                "Ocp-Apim-Subscription-Key": self._key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
                "User-Agent": "yassalam-tts-id",
            },
        )
        for attempt in range(4):
            try:
                with self._open(request, timeout=60) as response:
                    payload = response.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                    self._sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Azure TTS HTTP {exc.code}") from None
        self.characters_billed += len(text)
        audio, sr = sf.read(io.BytesIO(payload), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
