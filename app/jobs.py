"""Job processing: TTS (normalize -> chunk -> synthesize -> assemble) and transcription."""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import soundfile as sf

from app.audio import AudioBuilder
from app.config import Settings
from app.db import Database
from app.engines.base import TTSEngine, VoiceSpec
from app.engines.model_manager import ModelManager
from app.stt.whisper import WhisperSTT, to_srt
from app.text.chunker import chunk_text
from app.text.lexicon import Lexicon
from app.text.normalizer_id import apply_symbols, display_text, normalize, split_language_parts
from app.text.segmenter import split_segments
from app.worker import JobError

log = logging.getLogger(__name__)


def _pick(value: int | None, default: int) -> int:
    return default if value is None else int(value)


def plan_script(text: str, lexicon: Lexicon, max_chunk_chars: int) -> list[dict]:
    """Return the ordered render plan: {"kind": "speech", "text": chunk} or {"kind": "ayat", ...}."""
    lex = lexicon.get()
    plan: list[dict] = []
    for segment in split_segments(apply_symbols(text, lex)):
        if segment.kind == "ayat":
            plan.append({"kind": "ayat", "text": segment.text, "source": segment.source})
            continue
        for chunk in chunk_text(normalize(segment.text, lex), max_chunk_chars):
            plan.append({"kind": "speech", "text": display_text(chunk), "parts": split_language_parts(chunk, lex)})
    return plan


def render_calls(parts: list[tuple[str, str, str]], engine: TTSEngine) -> list[tuple[str, str | None]]:
    """Turn language parts into engine calls [(text, language)].

    Engines that can speak Arabic get separate calls for curated Arabic terms; Indonesian text right before a
    term ends with a comma so the voice keeps a continuing (not sentence-final) intonation. Other engines get
    one call with the Latin spellings.
    """
    if "ar" not in engine.info.extra_languages:
        return [(" ".join(fallback for _, _, fallback in parts), None)]
    calls: list[tuple[str, str | None]] = []
    for i, (lang, text, _) in enumerate(parts):
        if lang == "ar":
            calls.append((text, "ar"))
            continue
        followed_by_term = i + 1 < len(parts) and parts[i + 1][0] == "ar"
        if followed_by_term and text[-1] not in ".!?,;:…":
            text += ","
        calls.append((text, None))
    return calls


class JobProcessor:
    def __init__(self, settings: Settings, db: Database, engines: dict[str, TTSEngine],
                 manager: ModelManager, stt: WhisperSTT, lexicon: Lexicon):
        self.settings = settings
        self.db = db
        self.engines = engines
        self.manager = manager
        self.stt = stt
        self.lexicon = lexicon

    def job_dir(self, job_id: str) -> Path:
        path = self.settings.jobs_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __call__(self, job: dict, progress: Callable[[float], None]) -> dict:
        self.manager.release_fallbacks()
        if job["kind"] == "tts":
            return self._tts(job, progress)
        if job["kind"] == "transcribe":
            return self._transcribe(job, progress)
        raise JobError(f"unknown job kind {job['kind']!r}")

    def _voice(self, request: dict, engine: TTSEngine) -> VoiceSpec:
        reference = None
        if request.get("profile_id"):
            profile = self.db.get_profile(request["profile_id"])
            if profile is None:
                raise JobError("voice profile no longer exists")
            if profile["engine"] != engine.info.name:
                raise JobError(f"profile belongs to engine {profile['engine']!r}")
            reference = Path(profile["reference_path"])
        return VoiceSpec(voice=request.get("voice"), reference_path=reference, params=request.get("params") or {})

    def _tts(self, job: dict, progress: Callable[[float], None]) -> dict:
        request = job["request"]
        engine = self.engines.get(request["engine"])
        if engine is None:
            raise JobError(f"unknown engine {request['engine']!r}")
        ok, reason = engine.available()
        if not ok:
            raise JobError(reason)
        voice = self._voice(request, engine)
        plan = plan_script(request["text"], self.lexicon, self.settings.max_chunk_chars)
        speech_items = [p for p in plan if p["kind"] == "speech"]
        if not speech_items:
            raise JobError("nothing to synthesize after removing ayat blocks")

        crossfade = _pick(request.get("crossfade_ms"), self.settings.crossfade_ms)
        silence_ms = _pick(request.get("ayat_silence_ms"), self.settings.ayat_silence_ms)
        builder: AudioBuilder | None = None
        pending_silences: list[dict] = []
        synth_seconds = 0.0
        done = 0
        total = len(speech_items) + (1 if request.get("with_srt") else 0)

        for item in plan:
            if item["kind"] == "ayat":
                pending_silences.append(item)
                if builder is not None:
                    for s in pending_silences:
                        builder.add_silence(silence_ms, s["text"], s["source"])
                    pending_silences.clear()
                continue
            for text, language in render_calls(item["parts"], engine):
                spec = replace(voice, language=language) if language else voice
                t0 = time.monotonic()
                audio, sr = self.manager.run(engine, lambda e, t=text, v=spec: e.synthesize(t, v))
                synth_seconds += time.monotonic() - t0
                if builder is None:
                    builder = AudioBuilder(sample_rate=sr, crossfade_ms=crossfade)
                    for s in pending_silences:  # script starts with an ayat block
                        builder.add_silence(silence_ms, s["text"], s["source"])
                    pending_silences.clear()
                builder.add_speech(audio, sr)
            done += 1
            progress(done / total)

        assert builder is not None
        out_dir = self.job_dir(job["id"])
        audio_path = out_dir / "audio.wav"
        sf.write(audio_path, builder.build(), builder.sample_rate, subtype="PCM_16")

        result = {
            "engine": engine.info.name,
            "commercial_ok": engine.info.commercial_ok,
            "unofficial": engine.info.unofficial,
            "duration_s": round(builder.duration_s, 2),
            "sample_rate": builder.sample_rate,
            "chunks": len(speech_items),
            "characters": sum(len(p["text"]) for p in speech_items),
            "arabic_terms": sum(1 for p in speech_items for part in p["parts"] if part[0] == "ar"),
            "synth_seconds": round(synth_seconds, 2),
            "rtf": round(synth_seconds / builder.duration_s, 3) if builder.duration_s else None,
            "ayat_markers": [asdict(m) for m in builder.markers],
            "devices": self.manager.loaded,
        }
        if request.get("with_srt"):
            text, cues = self.manager.run(self.stt, lambda m: m.transcribe(audio_path, "id"))
            (out_dir / "subtitles.srt").write_text(to_srt(cues), encoding="utf-8")
            result["transcript"] = text
            progress(1.0)
        return result

    def _transcribe(self, job: dict, progress: Callable[[float], None]) -> dict:
        request = job["request"]
        source = Path(request["audio_path"])
        if not source.exists():
            raise JobError("uploaded audio is missing")
        text, cues = self.manager.run(self.stt, lambda m: m.transcribe(source, request.get("language", "id")))
        out_dir = self.job_dir(job["id"])
        (out_dir / "subtitles.srt").write_text(to_srt(cues), encoding="utf-8")
        progress(1.0)
        return {"transcript": text, "segments": len(cues), "devices": self.manager.loaded}
