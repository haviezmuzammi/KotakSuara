"""HTTP API. Every endpoint requires the `X-API-Key` header."""
from __future__ import annotations

import re
import secrets
import uuid
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from app.text.segmenter import AyatTagError, split_segments

ALLOWED_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    engine: str | None = None
    voice: str | None = None
    profile_id: str | None = None
    params: dict[str, float | int | str] = Field(default_factory=dict)
    ayat_silence_ms: int | None = Field(default=None, ge=0, le=60000)
    crossfade_ms: int | None = Field(default=None, ge=0, le=200)
    with_srt: bool = False

    @field_validator("profile_id")
    @classmethod
    def _hex_id(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[0-9a-f]{32}", v):
            raise ValueError("invalid profile_id")
        return v


def _state(request: Request):
    return request.app.state.svc


def require_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = _state(request).settings.api_key
    if not x_api_key or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


router = APIRouter(dependencies=[Depends(require_key)])


async def _save_upload(upload: UploadFile, target: Path) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO:
        raise HTTPException(422, f"audio must be one of {sorted(ALLOWED_AUDIO)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with target.open("wb") as fh:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                fh.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, "audio file too large (max 50 MB)")
            fh.write(chunk)


def _job_view(job: dict) -> dict:
    view = {k: job[k] for k in ("id", "kind", "status", "progress", "error", "result",
                                "created_at", "started_at", "finished_at")}
    view["engine"] = job["request"].get("engine")
    return view


@router.get("/health")
def health(request: Request) -> dict:
    svc = _state(request)
    return {
        "status": "ok",
        "busy": svc.worker.busy,
        "running_job": svc.worker.current_job,
        "queue_length": svc.db.queue_length(),
        "loaded_models": svc.manager.loaded,
        "device": svc.manager.preferred_device,
        "lexicon_error": svc.lexicon.last_error,
    }


@router.get("/v1/engines")
def engines(request: Request) -> list[dict]:
    out = []
    for engine in _state(request).engines.values():
        ok, reason = engine.available()
        out.append({**engine.info.as_dict(), "available": ok, "unavailable_reason": reason or None})
    return out


@router.post("/v1/tts", status_code=202)
def submit_tts(body: TTSRequest, request: Request) -> dict:
    svc = _state(request)
    engine_name = body.engine or svc.settings.default_engine
    engine = svc.engines.get(engine_name)
    if engine is None:
        raise HTTPException(422, f"unknown engine {engine_name!r}")
    ok, reason = engine.available()
    if not ok:
        raise HTTPException(422, f"engine {engine_name!r} unavailable: {reason}")
    if len(body.text) > svc.settings.max_text_chars:
        raise HTTPException(422, f"text longer than {svc.settings.max_text_chars} characters")
    try:
        segments = split_segments(body.text)
    except AyatTagError as exc:
        raise HTTPException(422, str(exc)) from None
    if not any(s.kind == "speech" for s in segments):
        raise HTTPException(422, "nothing to synthesize: text only contains ayat/Arabic blocks")
    if body.profile_id:
        profile = svc.db.get_profile(body.profile_id)
        if profile is None:
            raise HTTPException(404, "voice profile not found")
        if not engine.info.supports_clone or profile["engine"] != engine_name:
            raise HTTPException(422, f"profile is for engine {profile['engine']!r}")
    payload = body.model_dump()
    payload["engine"] = engine_name
    job_id = svc.db.create_job("tts", payload)
    svc.worker.notify()
    warnings = []
    if not engine.info.commercial_ok:
        warnings.append(f"{engine_name} is not licensed for commercial/monetized use")
    if engine.info.unofficial:
        warnings.append(f"{engine_name} uses an unofficial endpoint")
    return {"job_id": job_id, "status": "queued", "warnings": warnings}


@router.get("/v1/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    job = _state(request).db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return _job_view(job)


def _done_job_file(request: Request, job_id: str, name: str) -> Path:
    svc = _state(request)
    job = svc.db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    path = svc.settings.jobs_dir / job_id / name
    if not path.exists():
        raise HTTPException(404, f"{name} not available for this job")
    return path


@router.get("/v1/jobs/{job_id}/audio")
def job_audio(job_id: str, request: Request) -> FileResponse:
    return FileResponse(_done_job_file(request, job_id, "audio.wav"), media_type="audio/wav",
                        filename=f"{job_id}.wav")


@router.get("/v1/jobs/{job_id}/srt")
def job_srt(job_id: str, request: Request) -> PlainTextResponse:
    path = _done_job_file(request, job_id, "subtitles.srt")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="application/x-subrip")


@router.post("/v1/transcribe", status_code=202)
async def transcribe(request: Request, audio: Annotated[UploadFile, File()],
                     language: Annotated[str, Form(pattern=r"^[a-z]{2,3}$")] = "id") -> dict:
    svc = _state(request)
    target = svc.settings.data_dir / "uploads" / f"{uuid.uuid4().hex}{Path(audio.filename or '').suffix.lower()}"
    await _save_upload(audio, target)
    job_id = svc.db.create_job("transcribe", {"audio_path": str(target), "language": language})
    svc.worker.notify()
    return {"job_id": job_id, "status": "queued"}


@router.post("/v1/profiles", status_code=201)
async def create_profile(
    request: Request,
    name: Annotated[str, Form(min_length=1, max_length=100)],
    consent_by: Annotated[str, Form(min_length=2, max_length=200)],
    consent_date: Annotated[date, Form()],
    consent_evidence: Annotated[str, Form(min_length=5, max_length=1000)],
    reference_audio: Annotated[UploadFile, File()],
    engine: Annotated[str, Form()] = "chatterbox_ml",
    notes: Annotated[str, Form(max_length=1000)] = "",
) -> dict:
    """Consent fields are mandatory: a voice may only be cloned with its owner's recorded consent."""
    svc = _state(request)
    target_engine = svc.engines.get(engine)
    if target_engine is None or not target_engine.info.supports_clone:
        raise HTTPException(422, f"engine {engine!r} does not support voice cloning")
    if consent_date > date.today():
        raise HTTPException(422, "consent_date cannot be in the future")
    suffix = Path(reference_audio.filename or "").suffix.lower()
    target = svc.settings.profiles_dir / f"{uuid.uuid4().hex}{suffix}"
    await _save_upload(reference_audio, target)
    profile = svc.db.create_profile({
        "name": name.strip(), "engine": engine, "reference_path": str(target),
        "consent_by": consent_by.strip(), "consent_date": consent_date.isoformat(),
        "consent_evidence": consent_evidence.strip(), "notes": notes.strip(),
    })
    return _profile_view(profile)


def _profile_view(profile: dict) -> dict:
    return {k: v for k, v in profile.items() if k != "reference_path"}


@router.get("/v1/profiles")
def list_profiles(request: Request) -> list[dict]:
    return [_profile_view(p) for p in _state(request).db.list_profiles()]


@router.delete("/v1/profiles/{profile_id}", status_code=204)
def delete_profile(profile_id: str, request: Request) -> None:
    svc = _state(request)
    profile = svc.db.get_profile(profile_id)
    if profile is None:
        raise HTTPException(404, "voice profile not found")
    svc.db.delete_profile(profile_id)
    Path(profile["reference_path"]).unlink(missing_ok=True)


def build_api(svc) -> FastAPI:
    app = FastAPI(title="yassalam-tts-id", version="0.1.0", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.svc = svc
    app.include_router(router)
    return app

