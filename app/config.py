"""Service settings, read from environment variables with `.env` as fallback.

Environment variables win over `.env` so NSSM `AppEnvironmentExtra` can override anything.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    api_key: str
    azure_key: str = ""
    azure_region: str = ""
    host: str = "0.0.0.0"
    port: int = 8780
    data_dir: Path = REPO_ROOT / "data"
    lexicon_path: Path = REPO_ROOT / "config" / "lexicon.yaml"
    styles_path: Path = REPO_ROOT / "config" / "styles.yaml"
    ffmpeg_path: str = ""  # empty = ffmpeg from PATH; only needed for styles with an "fx" chain
    default_engine: str = "chatterbox_ml"
    stt_model: str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    max_loaded_models: int = 1
    max_chunk_chars: int = 250
    crossfade_ms: int = 50
    ayat_silence_ms: int = 2000
    max_text_chars: int = 40000
    stuck_job_seconds: int = 900
    enable_unofficial_engines: bool = True
    allowed_clients: tuple[str, ...] = ("127.0.0.1", "::1", "192.168.1.60")
    arabic_terms: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tts.sqlite3"


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(env_file: Path | None = None) -> Settings:
    env_file = env_file or Path(os.environ.get("YTTS_ENV_FILE", REPO_ROOT / ".env"))
    values: dict[str, str] = {}
    if env_file.exists():
        values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    values.update(os.environ)

    def get(name: str, default: str = "") -> str:
        return values.get(name, default).strip()

    api_key = get("API_KEY")
    if len(api_key) < 24:
        raise RuntimeError("API_KEY is missing or shorter than 24 characters; set it in .env")

    return Settings(
        api_key=api_key,
        azure_key=get("AZURE_SPEECH_KEY"),
        azure_region=get("AZURE_SPEECH_REGION"),
        host=get("HOST", "0.0.0.0"),
        port=int(get("PORT", "8780")),
        data_dir=Path(get("DATA_DIR", str(REPO_ROOT / "data"))),
        lexicon_path=Path(get("LEXICON_PATH", str(REPO_ROOT / "config" / "lexicon.yaml"))),
        styles_path=Path(get("STYLES_PATH", str(REPO_ROOT / "config" / "styles.yaml"))),
        ffmpeg_path=get("FFMPEG_PATH"),
        default_engine=get("DEFAULT_ENGINE", "chatterbox_ml"),
        stt_model=get("STT_MODEL", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
        max_loaded_models=int(get("MAX_LOADED_MODELS", "1")),
        max_chunk_chars=int(get("MAX_CHUNK_CHARS", "250")),
        crossfade_ms=min(200, max(0, int(get("CROSSFADE_MS", "50")))),
        ayat_silence_ms=int(get("AYAT_SILENCE_MS", "2000")),
        max_text_chars=int(get("MAX_TEXT_CHARS", "40000")),
        stuck_job_seconds=int(get("STUCK_JOB_SECONDS", "900")),
        enable_unofficial_engines=_bool(get("ENABLE_UNOFFICIAL_ENGINES", "true")),
        arabic_terms=_bool(get("ARABIC_TERMS", "true")),
        allowed_clients=tuple(a.strip() for a in get("ALLOWED_CLIENTS", "127.0.0.1,::1,192.168.1.60").split(",")
                              if a.strip()),
    )
